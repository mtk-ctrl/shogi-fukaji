// D23 research-only NNUE search. Existing independent legal/history core only.
// NNUE tensors come from the caller's verified snapshot; no classical leaf score.
#include "rules_state_ffi.cpp"
#include <cmath>
#include <iomanip>
#include <functional>
#include <atomic>
#include <new>

namespace {
constexpr size_t kNInputs=2548, kNHidden=32;
constexpr double kNMate=1000000.;
struct NModel { vector<float> w; array<double,kNHidden> b{},v{}; double o=0.; };
unordered_map<uint64_t,NModel> nnue_models;
uint64_t next_nnue_model=1;

int nnue_kind(const Piece& p) {
  const auto i=string("PLNSGBRK").find(p.type);
  if(i==string::npos)throw runtime_error("unknown NNUE piece");
  if(!p.promoted)return static_cast<int>(i)+1;
  switch(p.type) {
    case 'P':return 9; case 'L':return 10; case 'N':return 11;
    case 'S':return 12; case 'B':return 13; case 'R':return 14;
    default:throw runtime_error("invalid NNUE promotion");
  }
}
double nnue_value(const Board& b,const NModel& m,int perspective) {
  array<double,kNHidden> acc{};
  // Sum weights before bias, matching the Python full accumulator.
  const auto add=[&](size_t i) {
    if(i>=kNInputs)throw runtime_error("NNUE input range");
    for(size_t h=0;h<kNHidden;++h)acc[h]+=m.w[i*kNHidden+h];
  };
  for(int sq=0;sq<81;++sq) {
    const Piece& p=b.squares[sq];if(!p.type)continue;
    const int owner=(p.black?0:1)^perspective;
    const int relative=perspective?80-sq:sq;
    add(static_cast<size_t>((owner*14+nnue_kind(p)-1)*81+relative));
  }
  const auto hands=parse_hands(b.hand);
  for(int side=0;side<2;++side)for(int kind=0;kind<7;++kind)
    for(int j=0;j<min(20,hands[side][kind]);++j)
      add(static_cast<size_t>(2268+((side^perspective)*7+kind)*20+j));
  double value=0.;
  for(size_t h=0;h<kNHidden;++h)value+=clamp(acc[h]+m.b[h],0.,1.)*m.v[h];
  value+=m.o;
  if(!isfinite(value))throw runtime_error("nonfinite NNUE output");
  return value;
}

struct NTimeout {};
struct NSearch {
  const NModel& model;
  Board board;
  vector<string> history;
  vector<bool> mover_black,gave_check;
  unordered_map<string,string> hints;
  chrono::steady_clock::time_point deadline;
  const atomic<bool>* stop=nullptr;
  uint64_t nodes=0,evals=0;
  explicit NSearch(const NModel& m,const NativeState& s):model(m),board(s.board),
    history(s.history),mover_black(s.mover_black),gave_check(s.gave_check) {
    history.reserve(history.size()+64);mover_black.reserve(mover_black.size()+64);
    gave_check.reserve(gave_check.size()+64);
  }
  void time_check() const {if((stop&&stop->load(memory_order_relaxed))||chrono::steady_clock::now()>=deadline)throw NTimeout{};}
  vector<string> moves(bool use_hint=true) const {
    auto result=ordered_legal_moves(board);
    const auto score=[&](const string& word) {
      return research_piece_value(board.squares[from_usi_square(word[2],word[3])])+
             (word.size()==5?200:0);
    };
    stable_sort(result.begin(),result.end(),[&](const string& a,const string& b){return score(a)>score(b);});
    const auto hint=hints.find(history.back());
    if(use_hint&&hint!=hints.end())
      stable_sort(result.begin(),result.end(),[&](const string& a,const string& b){
        return (a==hint->second)>(b==hint->second);});
    return result;
  }
  // Status string and winner (-1 means draw/unresolved).
  pair<string,int> terminal(const vector<string>& legal) const {
    const int turn=board.turn=="b"?0:1;
    if(legal.empty())return checked(board,turn==0)?make_pair(string("checkmate"),1-turn):
                                                  make_pair(string("no_legal_unresolved"),-1);
    const string verdict=repetition_verdict(history,mover_black,gave_check);
    if(verdict=="draw")return {"repetition_draw",-1};
    if(verdict=="black_loses")return {"perpetual_check",1};
    if(verdict=="white_loses")return {"perpetual_check",0};
    return {"",-1};
  }
  struct MoveGuard {
    NSearch& search;Board saved;size_t h,m,c;
    MoveGuard(NSearch& s,const string& word):search(s),saved(s.board),h(s.history.size()),
      m(s.mover_black.size()),c(s.gave_check.size()) {
      // word belongs to this exact frame's independently generated legal list.
      try {
        const bool black=s.board.turn=="b";
        apply_validated_move(s.board,word);
        s.history.push_back(state_key(s.board));s.mover_black.push_back(black);
        s.gave_check.push_back(checked(s.board,s.board.turn=="b"));
      } catch(...) {s.board=std::move(saved);s.history.resize(h);s.mover_black.resize(m);s.gave_check.resize(c);throw;}
    }
    ~MoveGuard() noexcept {search.board=std::move(saved);search.history.resize(h);
      search.mover_black.resize(m);search.gave_check.resize(c);}
  };
  // D48 research-only strict mate proof, inspired by Kumoji MateSearch.
  // Attacker: only legal checking moves (OR). Defender: every legal reply (AND).
  // No external solutions, NNUE score, material scoring or opening knowledge.
  chrono::steady_clock::time_point mate_deadline=chrono::steady_clock::time_point::max();
  uint64_t mate_nodes=0, mate_root_guarded=0, mate_root_unsafe=0;
  vector<string> mate_unsafe_moves;
  string mate_proven_move;
  int mate_proven_plies=0;
  bool mate_incomplete=false;

  void mate_step() {
    ++mate_nodes;
    if((stop&&stop->load(memory_order_relaxed)) ||
       chrono::steady_clock::now()>=mate_deadline ||
       chrono::steady_clock::now()>=deadline)throw NTimeout{};
  }
  bool mate_attack(int remaining, bool root=false) {
    mate_step();
    if(remaining<=0) return false;
    // The game may have ended by repetition/continuous check, never call it mate.
    auto legal=ordered_legal_moves(board);
    if(!terminal(legal).first.empty())return false;
    for(const auto& move:legal) {
      mate_step();
      bool forced=false;
      {
        MoveGuard guard(*this,move);
        if(!checked(board,board.turn=="b"))continue; // legal but not a check
        forced=mate_defend(remaining-1);
      }
      if(forced) {
        if(root)mate_proven_move=move;
        return true;
      }
    }
    return false;
  }
  bool mate_defend(int remaining) {
    mate_step();
    // This side is the defender and is in check.
    const auto legal=ordered_legal_moves(board);
    const auto end=terminal(legal);
    if(end.first=="checkmate")return true;
    if(!end.first.empty() || remaining<=0)return false;
    for(const auto& move:legal) {
      mate_step();
      bool forced=false;
      {
        MoveGuard guard(*this,move);
        forced=mate_attack(remaining-1);
      }
      // A single legal escape which avoids force-mate disproves this attack.
      if(!forced)return false;
    }
    return true;
  }
  void mate_probe_offense(int milliseconds) {
    if(milliseconds<=0)return;
    const auto slice=chrono::steady_clock::now()+chrono::milliseconds(milliseconds);
    mate_deadline=min(deadline,slice);
    for(const int depth:{1,3,5}) {
      try {
        mate_proven_move.clear();
        if(mate_attack(depth,true)) {
          mate_proven_plies=depth;break;
        }
      }catch(const NTimeout&) {mate_incomplete=true;break;}
    }
    mate_deadline=chrono::steady_clock::time_point::max();
  }
  void mate_probe_defense_one(const vector<string>& root,int milliseconds) {
    if(milliseconds<=0 || mate_proven_plies>0 || root.size()<2)return;
    mate_deadline=min(deadline,chrono::steady_clock::now()+chrono::milliseconds(milliseconds));
    try {
      for(const auto& word:root) {
        mate_step();
        bool opponent_mate_one=false;
        {
          MoveGuard guard(*this,word);
          opponent_mate_one=mate_attack(1);
        }
        ++mate_root_guarded;
        if(opponent_mate_one) {
          mate_unsafe_moves.push_back(word);++mate_root_unsafe;
        }
      }
    }catch(const NTimeout&) {mate_incomplete=true;}
    mate_deadline=chrono::steady_clock::time_point::max();
  }

  double search(int depth,double alpha,double beta,int ply) {
    ++nodes;time_check();
    // Static leaves need only legal existence, not a complete sorted list.
    const auto legal=depth==0?pseudo_moves(board,true,true):moves();
    const auto end=terminal(legal);
    if(!end.first.empty())return end.second<0?0.:(kNMate-ply)*(end.second==(board.turn=="b"?0:1)?1.:-1.);
    if(depth==0) {++evals;return clamp(nnue_value(board,model,board.turn=="b"?0:1),-20.,20.)*1000.;}
    double best=-2*kNMate;string word;
    bool first_move=true;
    for(const auto& move:legal) {
      double score;
      {
        MoveGuard guard(*this,move);
        if(first_move) {
          score=-search(depth-1,-beta,-alpha,ply+1);
        } else {
          // Null-window probe is only a preliminary bound, never a skipped move.
          // The smallest representable positive window works for NNUE doubles.
          const double probe=std::nextafter(alpha,std::numeric_limits<double>::infinity());
          score=-search(depth-1,-probe,-alpha,ply+1);
          if(score>alpha && score<beta) {
            // A potentially best quiet/sacrifice move receives a full search.
            score=-search(depth-1,-beta,-alpha,ply+1);
          }
        }
      }
      first_move=false;
      if(score>best){best=score;word=move;}alpha=max(alpha,best);if(alpha>=beta)break;
    }
    if(!word.empty())hints[history.back()]=word;
    return best;
  }
};
}

extern "C" int fukaji_nnue_model_open(const float* w,size_t count,const double* b,
                                       const float* v,double o,uint64_t* output) noexcept {
  if(!w||!b||!v||!output||count!=kNInputs*kNHidden||!isfinite(o))return -1;
  try {
    lock_guard<mutex> lock(state_mutex);
    if(nnue_models.size()>=16)return -2;
    NModel model;model.w.assign(w,w+count);model.o=o;
    for(float value:model.w)if(!isfinite(value))return -1;
    for(size_t h=0;h<kNHidden;++h){if(!isfinite(b[h])||!isfinite(v[h]))return -1;model.b[h]=b[h];model.v[h]=v[h];}
    const uint64_t id=next_nnue_model++;nnue_models.emplace(id,std::move(model));*output=id;return 0;
  } catch(...) {return -3;}
}
extern "C" int fukaji_nnue_model_close(uint64_t model) noexcept {
  try {lock_guard<mutex> lock(state_mutex);return nnue_models.erase(model)==1?0:-1;}catch(...){return -3;}
}
extern "C" int fukaji_nnue_value(uint64_t model,uint64_t state,int perspective,double* output) noexcept {
  if(!output||perspective<0||perspective>1)return -1;
  try {lock_guard<mutex> lock(state_mutex);*output=nnue_value(find_state(state).board,nnue_models.at(model),perspective);return 0;}
  catch(...){return -3;}
}
static int nnue_search_impl(uint64_t model,uint64_t state,int milliseconds,int max_depth,
                            int fixed_depth,char* output,size_t capacity,const atomic<bool>* stop) noexcept {
  if(milliseconds<1||milliseconds>10000||max_depth<1||max_depth>64||
     (fixed_depth!=0&&fixed_depth!=1))return -1;
  try {
    lock_guard<mutex> lock(state_mutex);
    const auto started=chrono::steady_clock::now();
    NSearch search(nnue_models.at(model),find_state(state));
    search.stop=stop;
    search.deadline=fixed_depth?chrono::steady_clock::time_point::max():started+chrono::milliseconds(milliseconds);
    auto moves=search.moves(false);const auto end=search.terminal(moves);
    string best=moves.empty()||!end.first.empty()?"":moves[0];
    int completed=0,partial=0;double score=0.;bool has_score=false;
    bool research_mate_proven=false;
    if(!fixed_depth && !best.empty() && end.first.empty()) {
      search.mate_probe_offense(milliseconds>=1000?70:max(1,milliseconds/9));
      if(search.mate_proven_plies>0 && !search.mate_proven_move.empty()) {
        best=search.mate_proven_move;
        completed=search.mate_proven_plies;
        score=kNMate-completed;
        has_score=true;
        research_mate_proven=true;
      } else {
        search.mate_probe_defense_one(moves,milliseconds>=1000?24:max(1,milliseconds/25));
        if(!search.mate_unsafe_moves.empty() && search.mate_unsafe_moves.size()<moves.size()) {
          moves.erase(remove_if(moves.begin(),moves.end(),[&](const string& word) {
            return find(search.mate_unsafe_moves.begin(),search.mate_unsafe_moves.end(),word)!=
                   search.mate_unsafe_moves.end();
          }),moves.end());
          best=moves.front();
        }
      }
    }
    if(!research_mate_proven)for(int depth=1;depth<=max_depth&&!best.empty();++depth) {
      string current=best;double value=-2*kNMate,alpha=-2*kNMate;int searched=0;
      try {
        for(const auto& move:moves) {
          search.time_check();double v;
          {NSearch::MoveGuard guard(search,move);v=-search.search(depth-1,-2*kNMate,-alpha,1);}
          if(v>value){value=v;current=move;}alpha=max(alpha,value);++searched;
        }
      }catch(const NTimeout&){if(completed==0&&searched){best=current;score=value;has_score=true;partial=searched;}break;}
      best=current;score=value;has_score=true;completed=depth;
      stable_sort(moves.begin(),moves.end(),[&](const string& a,const string& b){return (a==best)>(b==best);});
      if(abs(score)>kNMate-1000.)break;
    }
    if(state_key(search.board)!=find_state(state).history.back()||search.history!=find_state(state).history)
      throw runtime_error("native search history restoration failed");
    const double elapsed=chrono::duration<double,milli>(chrono::steady_clock::now()-started).count();
    ostringstream out;out<<setprecision(17)<<"{\"move\":";
    if(best.empty())out<<"null";else out<<'"'<<best<<'"';
    out<<",\"depth\":"<<completed<<",\"partial_root_moves\":"<<partial<<",\"root_moves\":"<<moves.size()
       <<",\"nodes\":"<<search.nodes<<",\"evaluations\":"<<search.evals<<",\"score\":";
    if(has_score)out<<score;else out<<"null";
    out<<",\"elapsed_ms\":"<<elapsed<<",\"quiescence_depth\":0,\"quiescence_nodes\":0,\"terminal\":";
    if(end.first.empty())out<<"null";else {out<<"{\"status\":\""<<end.first<<"\",\"winner\":";
      if(end.second<0)out<<"null";else out<<end.second;out<<'}';}
    out<<",\"research_mate_proven\":"<<(research_mate_proven?"true":"false")
       <<",\"research_mate_plies\":"<<search.mate_proven_plies
       <<",\"research_mate_nodes\":"<<search.mate_nodes
       <<",\"research_mate_incomplete\":"<<(search.mate_incomplete?"true":"false")
       <<",\"research_defense_checked\":"<<search.mate_root_guarded
       <<",\"research_proven_unsafe\":"<<search.mate_root_unsafe;
    out<<'}';return copy_text(out.str(),output,capacity);
  }catch(...){return -3;}
}
extern "C" int fukaji_nnue_search(uint64_t model,uint64_t state,int milliseconds,int max_depth,
                                   int fixed_depth,char* output,size_t capacity) noexcept {
  return nnue_search_impl(model,state,milliseconds,max_depth,fixed_depth,output,capacity,nullptr);
}
// Cancellation never takes state_mutex, which is held by the search thread.
// The caller owns this token and frees it only after the search has returned.
extern "C" void* fukaji_nnue_stop_open() noexcept {return new(nothrow) atomic<bool>(false);}
extern "C" void fukaji_nnue_stop_signal(void* token) noexcept {
  if(token)static_cast<atomic<bool>*>(token)->store(true,memory_order_relaxed);
}
extern "C" void fukaji_nnue_stop_close(void* token) noexcept {delete static_cast<atomic<bool>*>(token);}
extern "C" int fukaji_nnue_search_interruptible(uint64_t model,uint64_t state,int milliseconds,
    int max_depth,int fixed_depth,char* output,size_t capacity,void* token) noexcept {
  if(!token)return -1;
  return nnue_search_impl(model,state,milliseconds,max_depth,fixed_depth,output,capacity,
                         static_cast<atomic<bool>*>(token));
}
