// Research-only same-process C ABI. Reuses the independent native rule core;
// never linked to the production USI engine. No external code or dynamic loading.
#define main fukaji_rules_probe_standalone_main
#include "rules_probe.cpp"
#undef main
#include <cstring>
static int legal_impl(const char* sfen, char* output, size_t capacity,
                      bool pseudo_order) noexcept {
  if (!sfen || !output || capacity == 0 || capacity > 65536) return -1;
  try {
    const string input(sfen);
    if (input.size() > 1024 || input.find('\n') != string::npos) return -1;
    Board board = parse(input);
    (void)parse_hands(board.hand);
    const string moves = encode_moves(pseudo_order
        ? ordered_legal_moves(board) : pseudo_moves(board, true));
    if (moves.size() + 1 > capacity) return -2;
    memcpy(output, moves.c_str(), moves.size() + 1);
    return static_cast<int>(moves.size());
  } catch (...) {
    return -3;
  }
}
extern "C" int fukaji_legal_usi(const char* sfen, char* output,
                                 size_t capacity) noexcept {
  return legal_impl(sfen, output, capacity, false);
}
extern "C" int fukaji_legal_pseudo_order_usi(const char* sfen, char* output,
                                              size_t capacity) noexcept {
  return legal_impl(sfen, output, capacity, true);
}
