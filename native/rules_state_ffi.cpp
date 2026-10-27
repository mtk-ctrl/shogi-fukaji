// Research-only, opt-in stateful C ABI. Not linked to the production USI engine.
// Opaque numeric handles avoid dereferencing arbitrary caller-provided pointers.
// A state is capped at 512 reversible moves; at most 16 simultaneous states.
// All operations are bounded and synchronized; failures leave states unchanged.
#define main fukaji_rules_probe_standalone_main
#include "rules_probe.cpp"
#undef main
#include <cstring>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

namespace {
struct LegalCache {
  // Exact search-frame identity; identical SFEN alone cannot reuse a cache.
  vector<string> moves;
  string position_key;
  uint64_t frame_token=0;
  size_t history_length=0;
  int ply=0;
  bool valid=false;
};
struct NativeState {
  Board board;
  vector<Board> undo;
  vector<string> history;
  vector<bool> mover_black;
  vector<bool> gave_check;
  vector<LegalCache> legal_cache;
  vector<uint64_t> frame_tokens;
  uint64_t next_frame_token=2;
  uint64_t cache_hits=0, fallback_pushes=0, legal_generations=0;
  uint64_t validate_ns=0, apply_ns=0, history_ns=0;
};
mutex state_mutex;
unordered_map<uint64_t,NativeState> state_store;
uint64_t next_handle=1;
constexpr size_t kMaxInput=1024;
constexpr size_t kMaxStates=16;
constexpr size_t kMaxUndo=512;

string bounded_input(const char* data,size_t length,size_t maximum) {
  if(!data || length==0 || length>maximum)throw runtime_error("input length");
  string result(data,length);
  if(result.find('\0')!=string::npos || result.find('\n')!=string::npos ||
     result.find('\r')!=string::npos)throw runtime_error("control character");
  return result;
}
int copy_text(const string& value,char* out,size_t capacity) {
  if(!out || capacity==0 || capacity>65536)return -1;
  if(value.size()+1>capacity)return -2;
  memcpy(out,value.c_str(),value.size()+1);
  return static_cast<int>(value.size());
}
NativeState& find_state(uint64_t handle) {
  auto it=state_store.find(handle);
  if(it==state_store.end())throw runtime_error("unknown handle");
  return it->second;
}
}

extern "C" int fukaji_state_open(const char* sfen,size_t length,
                                  uint64_t* handle) noexcept {
  if(!handle)return -1;
  *handle=0;
  try {
    const string input=bounded_input(sfen,length,kMaxInput);
    Board board=parse(input);
    board.hand=encode_hand(parse_hands(board.hand));
    if(board.kings[0]<0 || board.kings[1]<0)return -1;
    NativeState state;
    state.board=board;
    // Preallocate bounded history so successful push/pop cannot allocate after
    // the new board has been validated. Failure during open is transactional.
    state.undo.reserve(kMaxUndo);
    state.history.reserve(kMaxUndo+1);
    state.mover_black.reserve(kMaxUndo);
    state.gave_check.reserve(kMaxUndo);
    state.frame_tokens.reserve(kMaxUndo+1);
    state.frame_tokens.push_back(1);
    state.legal_cache.resize(kMaxUndo+1);
    state.history.push_back(state_key(board));
    lock_guard<mutex> lock(state_mutex);
    if(state_store.size()>=kMaxStates ||
       next_handle==numeric_limits<uint64_t>::max())return -2;
    const uint64_t id=next_handle++;
    state_store.emplace(id,std::move(state));
    *handle=id;
    return 0;
  } catch (...) {return -3;}
}

extern "C" int fukaji_state_close(uint64_t handle) noexcept {
  lock_guard<mutex> lock(state_mutex);
  return state_store.erase(handle)==1?0:-1;
}

extern "C" int fukaji_state_legal(uint64_t handle,char* output,
                                   size_t capacity) noexcept {
  try {
    lock_guard<mutex> lock(state_mutex);
    NativeState& state=find_state(handle);
    LegalCache& cache=state.legal_cache[state.undo.size()];
    const bool valid=cache.valid &&
      cache.frame_token==state.frame_tokens.back() &&
      cache.history_length==state.history.size() &&
      cache.ply==state.board.ply &&
      cache.position_key==state.history.back();
    if(valid)return copy_text(encode_moves(cache.moves),output,capacity);
    vector<string> moves=ordered_legal_moves(state.board);
    const int rc=copy_text(encode_moves(moves),output,capacity);
    if(rc>=0) {
      // No partial cache publication on allocation or output failure.
      string key=state.history.back();
      cache.moves=std::move(moves);
      cache.position_key=std::move(key);
      cache.frame_token=state.frame_tokens.back();
      cache.history_length=state.history.size();
      cache.ply=state.board.ply;
      cache.valid=true;
      ++state.legal_generations;
    }
    return rc;
  } catch (...) {return -3;}
}

// Research-only native capture/promotion ordering.  The validated cache
// retains pseudo-legal iteration order; this function returns a stable
// score-ordered copy, matching fukaji.engine.ordered_moves exactly.
static int research_piece_value(const Piece& piece) {
  if(!piece.type)return 0;
  if(piece.promoted) {
    switch(piece.type) {
      case 'P':return 500;
      case 'L':return 520;
      case 'N':return 620;
      case 'S':return 850;
      case 'B':return 1100;
      case 'R':return 650;
    }
  }
  switch(piece.type) {
    case 'P':return 100;
    case 'L':return 300;
    case 'N':return 320;
    case 'S':return 480;
    case 'G':return 760;
    case 'B':return 900;
    case 'R':return 600;
    case 'K':return 0;
  }
  throw runtime_error("unexpected capture piece");
}
extern "C" int fukaji_state_scored(uint64_t handle,char* output,
                                    size_t capacity) noexcept {
  try {
    lock_guard<mutex> lock(state_mutex);
    NativeState& state=find_state(handle);
    LegalCache& cache=state.legal_cache[state.undo.size()];
    const bool valid=cache.valid &&
      cache.frame_token==state.frame_tokens.back() &&
      cache.history_length==state.history.size() &&
      cache.ply==state.board.ply &&
      cache.position_key==state.history.back();
    vector<string> pseudo=valid?cache.moves:ordered_legal_moves(state.board);
    vector<string> scored=pseudo;
    const auto score=[&](const string& move) {
      const int to=from_usi_square(move[2],move[3]);
      return research_piece_value(state.board.squares[to])+
             (move.size()==5?200:0);
    };
    stable_sort(scored.begin(),scored.end(),
                [&](const string& a,const string& b){return score(a)>score(b);});
    const int rc=copy_text(encode_moves(scored),output,capacity);
    if(rc>=0 && !valid) {
      // Only successful output publishes a frame-validated legal cache.
      string key=state.history.back();
      cache.moves=std::move(pseudo);
      cache.position_key=std::move(key);
      cache.frame_token=state.frame_tokens.back();
      cache.history_length=state.history.size();
      cache.ply=state.board.ply;
      cache.valid=true;
      ++state.legal_generations;
    }
    return rc;
  } catch (...) {return -3;}
}

// Research-only compact legal move transport. A 32-bit code stores the
// python-shogi square indices (rank-major), promotion, and optional drop kind.
// It does not grant legality: state_push still checks the exact frame cache.
// No output or cache is published on invalid capacity or packing failure.
extern "C" int fukaji_state_scored_packed(uint64_t handle,uint32_t* output,
                                          size_t capacity) noexcept {
  try {
    if(!output || capacity==0 || capacity>1024)return -1;
    lock_guard<mutex> lock(state_mutex);
    NativeState& state=find_state(handle);
    LegalCache& cache=state.legal_cache[state.undo.size()];
    const bool valid=cache.valid &&
      cache.frame_token==state.frame_tokens.back() &&
      cache.history_length==state.history.size() &&
      cache.ply==state.board.ply &&
      cache.position_key==state.history.back();
    vector<string> pseudo=valid?cache.moves:ordered_legal_moves(state.board);
    vector<string> scored=pseudo;
    const auto score=[&](const string& move) {
      const int to=from_usi_square(move[2],move[3]);
      return research_piece_value(state.board.squares[to])+
             (move.size()==5?200:0);
    };
    stable_sort(scored.begin(),scored.end(),
                [&](const string& a,const string& b){return score(a)>score(b);});
    if(scored.size()>capacity)return -2;
    const auto python_square=[](char file,char rank)->uint32_t {
      if(file<'1'||file>'9'||rank<'a'||rank>'i')
        throw runtime_error("invalid packed square");
      return static_cast<uint32_t>((rank-'a')*9+('9'-file));
    };
    vector<uint32_t> packed;
    packed.reserve(scored.size());
    for(const string& move:scored) {
      if(move.size()!=4 && move.size()!=5)
        throw runtime_error("invalid generated move size");
      const uint32_t to=python_square(move[2],move[3]);
      uint32_t from=127,drop=0,promote=0;
      if(move[1]=='*') {
        if(move.size()!=4)throw runtime_error("invalid generated drop");
        const auto index=string("PLNSGBR").find(move[0]);
        if(index==string::npos)throw runtime_error("invalid drop kind");
        drop=static_cast<uint32_t>(index+1);
      } else {
        from=python_square(move[0],move[1]);
        promote=move.size()==5?1:0;
        if(promote && move[4]!='+')throw runtime_error("invalid promotion");
      }
      packed.push_back(to|(from<<7)|(promote<<14)|(drop<<15));
    }
    // Prepare all potentially throwing state before publishing the cache.
    string key=valid?string():state.history.back();
    if(!valid) {
      cache.moves=std::move(pseudo);
      cache.position_key=std::move(key);
      cache.frame_token=state.frame_tokens.back();
      cache.history_length=state.history.size();
      cache.ply=state.board.ply;
      cache.valid=true;
      ++state.legal_generations;
    }
    for(size_t i=0;i<packed.size();++i)output[i]=packed[i];
    return static_cast<int>(packed.size());
  } catch (...) {return -3;}
}

extern "C" int fukaji_state_sfen(uint64_t handle,char* output,
                                  size_t capacity) noexcept {
  try {
    lock_guard<mutex> lock(state_mutex);
    return copy_text(format_sfen(find_state(handle).board),output,capacity);
  } catch (...) {return -3;}
}

extern "C" int fukaji_state_verdict(uint64_t handle,char* output,
                                     size_t capacity) noexcept {
  try {
    lock_guard<mutex> lock(state_mutex);
    const NativeState& state=find_state(handle);
    return copy_text(repetition_verdict(state.history,state.mover_black,
                                        state.gave_check),output,capacity);
  } catch (...) {return -3;}
}

extern "C" int fukaji_state_is_check(uint64_t handle) noexcept {
  try {
    lock_guard<mutex> lock(state_mutex);
    const Board& board=find_state(handle).board;
    return checked(board,board.turn=="b")?1:0;
  } catch (...) {return -3;}
}

// Read-only research evaluation snapshot. Layout: 81 rank-major squares
// (signed python-shogi piece type; positive=black), 7 black hand counts,
// 7 white hand counts, and side-to-move (0=black, 1=white).
// A snapshot is NOT a legality certificate and never changes the state.
extern "C" int fukaji_state_eval_snapshot(uint64_t handle,int16_t* output,
                                          size_t capacity) noexcept {
  constexpr size_t kWords=96;
  if(!output || capacity<kWords || capacity>1024)return -1;
  try {
    lock_guard<mutex> lock(state_mutex);
    const Board& b=find_state(handle).board;
    array<int16_t,kWords> values{};
    const string types="PLNSGBRK";
    for(size_t sq=0;sq<81;++sq) {
      const Piece& p=b.squares[sq];
      if(!p.type)continue;
      const auto idx=types.find(p.type);
      if(idx==string::npos)throw runtime_error("invalid snapshot piece");
      int code=static_cast<int>(idx)+1;
      if(p.promoted) {
        switch(p.type) {
          case 'P':code=9;break;
          case 'L':code=10;break;
          case 'N':code=11;break;
          case 'S':code=12;break;
          case 'B':code=13;break;
          case 'R':code=14;break;
          default:throw runtime_error("invalid promoted snapshot piece");
        }
      }
      values[sq]=static_cast<int16_t>(p.black?code:-code);
    }
    const auto hands=parse_hands(b.hand);
    for(size_t side=0;side<2;++side)
      for(size_t kind=0;kind<7;++kind)
        values[81+side*7+kind]=static_cast<int16_t>(hands[side][kind]);
    values[95]=static_cast<int16_t>(b.turn=="b"?0:1);
    memcpy(output,values.data(),sizeof(values));
    return static_cast<int>(kWords);
  } catch (...) {return -3;}
}

// A generated legal list is valid only for its exact frame and history.
// Parent caches survive child push/pop. Missing caches use full validation.
extern "C" int fukaji_state_push(uint64_t handle,const char* usi,
                                  size_t length) noexcept {
  try {
    const string move=bounded_input(usi,length,5);
    lock_guard<mutex> lock(state_mutex);
    NativeState& state=find_state(handle);
    if(state.undo.size()>=kMaxUndo ||
       state.next_frame_token==numeric_limits<uint64_t>::max())return -2;
    const bool moved_black=state.board.turn=="b";
    Board next=state.board;
    const auto started=chrono::steady_clock::now();
    const LegalCache& cache=state.legal_cache[state.undo.size()];
    const bool cached=cache.valid &&
      cache.frame_token==state.frame_tokens.back() &&
      cache.history_length==state.history.size() &&
      cache.ply==state.board.ply &&
      cache.position_key==state.history.back();
    if(cached) {
      // A cache hit is not an unchecked push: membership is mandatory.
      if(find(cache.moves.begin(),cache.moves.end(),move)==cache.moves.end())
        return -2;
    } else {
      const vector<string> legal=pseudo_moves(state.board,true);
      if(!binary_search(legal.begin(),legal.end(),move))return -2;
    }
    const auto validated=chrono::steady_clock::now();
    apply_validated_move(next,move);
    const auto applied=chrono::steady_clock::now();
    const bool gave_check=checked(next,next.turn=="b");
    string key=state_key(next);
    // All throwing work is complete before committing bounded state.
    state.undo.push_back(state.board);
    state.history.push_back(std::move(key));
    state.mover_black.push_back(moved_black);
    state.gave_check.push_back(gave_check);
    state.board=std::move(next);
    state.frame_tokens.push_back(state.next_frame_token++);
    state.legal_cache[state.undo.size()].valid=false;
    if(cached)++state.cache_hits;
    else ++state.fallback_pushes;
    const auto finished=chrono::steady_clock::now();
    state.validate_ns+=static_cast<uint64_t>(
      chrono::duration_cast<chrono::nanoseconds>(validated-started).count());
    state.apply_ns+=static_cast<uint64_t>(
      chrono::duration_cast<chrono::nanoseconds>(applied-validated).count());
    state.history_ns+=static_cast<uint64_t>(
      chrono::duration_cast<chrono::nanoseconds>(finished-applied).count());
    return 0;
  } catch (...) {return -3;}
}

extern "C" int fukaji_state_pop(uint64_t handle) noexcept {
  try {
    lock_guard<mutex> lock(state_mutex);
    NativeState& state=find_state(handle);
    if(state.undo.empty())return -2;
    state.board=std::move(state.undo.back());
    state.undo.pop_back();
    state.frame_tokens.pop_back();
    state.history.pop_back();
    state.mover_black.pop_back();
    state.gave_check.pop_back();
    return 0;
  } catch (...) {return -3;}
}

extern "C" int fukaji_state_profile(uint64_t handle,char* output,
                                     size_t capacity) noexcept {
  try {
    lock_guard<mutex> lock(state_mutex);
    const NativeState& state=find_state(handle);
    return copy_text(
      "cache_hits="+to_string(state.cache_hits)+
      ";fallback_pushes="+to_string(state.fallback_pushes)+
      ";legal_generations="+to_string(state.legal_generations)+
      ";validate_ns="+to_string(state.validate_ns)+
      ";apply_ns="+to_string(state.apply_ns)+
      ";history_ns="+to_string(state.history_ns),
      output,capacity);
  } catch (...) {return -3;}
}
