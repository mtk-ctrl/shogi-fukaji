// FUKAJI NNUE leaf evaluation for the KUMOJI search (hybrid research engine).
// Only the leaf score is replaced; search, ordering, TT and mate code are KUMOJI's.
// Weights: raw file written by scripts/hybrid_export_weights.py (see layout there).
// Env: FUKAJI_NNUE_FILE (required to enable), FUKAJI_NNUE_SCALE (centipawns per logit, default 600).
#pragma once
#include "rules/position.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace shogi::strategy {
namespace nnue_hybrid {
constexpr int kBaseInputs = 2548, kKingInputs = 17576, kHidden = 128;
struct Model {
    bool loaded = false, tried = false, king_relative = false;
    int inputs = kBaseInputs;
    std::vector<float> w;                 // [inputs][kHidden] (inputs = 2548, or 17576 with king-relative rows)
    std::array<float, kHidden> b{}, v{};
    float o = 0.f;
    double scale = 600.0;
};
inline Model& model() { static Model m; return m; }
inline void load_once() {
    Model& m = model();
    if (m.tried) return;
    m.tried = true;
    const char* path = std::getenv("FUKAJI_NNUE_FILE");
    if (!path || !*path) return;
    if (const char* s = std::getenv("FUKAJI_NNUE_SCALE")) m.scale = std::atof(s);
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "FUKAJI_NNUE_FILE unreadable: %s\n", path); std::abort(); }
    char magic[8];
    std::uint32_t inputs = 0, hidden = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::fread(&inputs, 4, 1, f) == 1 && std::fread(&hidden, 4, 1, f) == 1 &&
              std::string(magic, 8) == "FKJNNUE1" && (inputs == kBaseInputs || inputs == kKingInputs) && hidden == kHidden;
    if (ok) {
        m.inputs = int(inputs);
        m.king_relative = inputs == kKingInputs;
        m.w.resize(std::size_t(m.inputs) * kHidden);
        ok = std::fread(m.w.data(), 4, m.w.size(), f) == m.w.size() &&
             std::fread(m.b.data(), 4, kHidden, f) == kHidden && std::fread(m.v.data(), 4, kHidden, f) == kHidden &&
             std::fread(&m.o, 4, 1, f) == 1;
    }
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "FUKAJI_NNUE_FILE format error: %s\n", path); std::abort(); }
    m.loaded = true;
}
inline bool enabled() { load_once(); return model().loaded; }

// KUMOJI kind (1 P,2 L,3 N,4 S,5 B,6 R,7 G,8 K) -> FUKAJI NNUE kind (1 P,2 L,3 N,4 S,5 G,6 B,7 R,8 K, 9..14 promoted P L N S B R).
inline int nnue_kind(int kind, bool promoted) {
    if (promoted) {
        switch (kind) { case 1: return 9; case 2: return 10; case 3: return 11; case 4: return 12; case 5: return 13; case 6: return 14; default: break; }
    }
    switch (kind) { case 1: return 1; case 2: return 2; case 3: return 3; case 4: return 4; case 7: return 5; case 5: return 6; case 6: return 7; case 8: return 8; default: return 0; }
}
// Logit for the side to move (same as fukaji nnue_value with perspective = side to move).
inline double logit_side_to_move(const rules::Snapshot& s) {
    const Model& m = model();
    const int persp = s.turn == rules::Color::Black ? 0 : 1;
    float acc[kHidden] = {};
    auto add = [&](int idx) {
        const float* row = &m.w[std::size_t(idx) * kHidden];
        for (int h = 0; h < kHidden; ++h) acc[h] += row[h];
    };
    int kings[2] = {-1, -1};                                  // python-shogi squares of the Black/White king
    for (int sq = 0; sq < 81; ++sq) {
        const auto& p = s.board[sq];
        if (p.kind == 8) kings[p.color == rules::Color::Black ? 0 : 1] = (sq % 9) * 9 + (9 - (sq / 9 + 1));
    }
    for (int sq = 0; sq < 81; ++sq) {
        const auto& p = s.board[sq];
        if (!p.kind) continue;
        const int file = sq / 9 + 1, rank = sq % 9;       // KUMOJI index = (file-1)*9 + rank
        const int py = rank * 9 + (9 - file);               // python-shogi / FUKAJI square (0 = 9a)
        const int owner = (p.color == rules::Color::Black ? 0 : 1) ^ persp;
        const int relative = persp ? 80 - py : py;
        add((owner * 14 + nnue_kind(p.kind, p.promoted) - 1) * 81 + relative);
        // King-relative rows (fukaji.nnue_research.relative_piece_indices): anchor = own king, then enemy king.
        if (m.king_relative && p.kind != 8) {
            const int nk = nnue_kind(p.kind, p.promoted);
            const int kind = nk - 1 - (nk > 8 ? 1 : 0);
            const int direction = persp == 0 ? 1 : -1;
            for (int anchor = 0; anchor < 2; ++anchor) {
                const int king = kings[anchor == 0 ? persp : 1 - persp];
                if (king < 0) continue;
                const int row = direction * (py / 9 - king / 9) + 8, col = direction * (py % 9 - king % 9) + 8;
                add(kBaseInputs + (((anchor * 2 + owner) * 13 + kind) * 17 + row) * 17 + col);
            }
        }
    }
    // KUMOJI hands order: P L N S B R G ; FUKAJI hand kind index: P L N S G B R (0..6).
    static constexpr int map_hand[7] = {0, 1, 2, 3, 5, 6, 4};
    for (int color = 0; color < 2; ++color)
        for (int k = 0; k < 7; ++k) {
            const int n = std::min(20, s.hands[color][k]);
            const int side = color;                            // 0 black, 1 white (python color)
            for (int j = 0; j < n; ++j) add(2268 + ((side ^ persp) * 7 + map_hand[k]) * 20 + j);
        }
    float value = m.o;
    for (int h = 0; h < kHidden; ++h) value += std::clamp(acc[h] + m.b[h], 0.f, 1.f) * m.v[h];
    return value;
}
inline int eval_black(const rules::Snapshot& s) {
    const double l = std::clamp(logit_side_to_move(s), -20.0, 20.0);
    const double cp = l * model().scale;
    return int(s.turn == rules::Color::Black ? cp : -cp);
}
} // namespace nnue_hybrid
} // namespace shogi::strategy
