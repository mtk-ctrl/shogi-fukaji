// Reads SFEN lines on stdin, prints "<logit_side_to_move> <eval_black_cp>" for the hybrid NNUE (verification only).
#include "rules/position.h"
#include "strategy/nnue_eval.h"
#include <iostream>
int main() {
    using namespace shogi;
    if (!strategy::nnue_hybrid::enabled()) { std::cerr << "set FUKAJI_NNUE_FILE\n"; return 2; }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        rules::Position pos; std::string err;
        if (!pos.set(line, {}, err)) { std::cout << "ERR " << err << "\n"; continue; }
        const auto s = pos.snapshot();
        std::cout.precision(9);
        std::cout << strategy::nnue_hybrid::logit_side_to_move(s) << " " << strategy::nnue_hybrid::eval_black(s) << "\n";
    }
}
