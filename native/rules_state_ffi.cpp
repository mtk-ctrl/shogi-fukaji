// Research-only, opt-in stateful C ABI. Not linked to the production USI engine.
// Opaque numeric handles avoid dereferencing arbitrary caller-provided pointers.
// A state is capped at 512 reversible moves; at most 16 simultaneous states.
// All operations are bounded and synchronized; failures leave states unchanged.
#define main fukaji_rules_probe_standalone_main
#include "rules_probe.cpp"
#undef main
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

namespace {
struct NativeState {
  Board board;
  vector<Board> undo;
  vector<string> history;
  vector<bool> mover_black;
  vector<bool> gave_check;
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
    return copy_text(encode_moves(ordered_legal_moves(find_state(handle).board)),
                     output,capacity);
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

// Only the current Board is snapshotted per move. Histories are append-only
// bounded vectors; no NativeState-wide copy on either push or pop.
// Every potentially throwing operation precedes the first state mutation.
extern "C" int fukaji_state_push(uint64_t handle,const char* usi,
                                  size_t length) noexcept {
  try {
    const string move=bounded_input(usi,length,5);
    lock_guard<mutex> lock(state_mutex);
    NativeState& state=find_state(handle);
    if(state.undo.size()>=kMaxUndo)return -2;
    const bool moved_black=state.board.turn=="b";
    Board next=state.board;
    apply_legal_move(next,move);
    const bool gave_check=checked(next,next.turn=="b");
    string key=state_key(next);
    // The only potentially throwing state mutation. std::vector::push_back
    // preserves its original contents if Board copy construction throws.
    state.undo.push_back(state.board);
    // All capacities were reserved in state_open. Moving string/Board and
    // appending bools cannot throw with the standard C++17 allocators.
    state.history.push_back(std::move(key));
    state.mover_black.push_back(moved_black);
    state.gave_check.push_back(gave_check);
    state.board=std::move(next);
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
    state.history.pop_back();
    state.mover_black.pop_back();
    state.gave_check.pop_back();
    return 0;
  } catch (...) {return -3;}
}
