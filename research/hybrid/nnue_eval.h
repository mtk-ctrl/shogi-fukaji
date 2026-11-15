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
constexpr int kBaseInputs = 2548, kKingInputs = 17576, kZoneInputs = 3848, kHidden = 128, kMaxHidden = 256;   // hidden: 128(正式) または 256(研究)
struct Model {
    bool loaded = false, tried = false, king_relative = false, king_zone = false;
    int inputs = kBaseInputs, hidden = kHidden;
    std::vector<float> w;                 // [inputs][hidden] (inputs = 2548, or 17576 with king-relative rows)
    std::array<float, kMaxHidden> b{}, v{};
    float o = 0.f;
    // G2(研究): 両視点＋2層目。評価 = 従来の式 + w3・clip(L2 [自の中間層, 相手の中間層] + b2)。h2 == 0 なら従来どおり。
    int h2 = 0;
    std::vector<float> l2;                // [2*hidden][h2]
    std::array<float, 64> b2{}, w3{};
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
    std::uint32_t inputs = 0, hidden = 0, h2 = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::fread(&inputs, 4, 1, f) == 1 && std::fread(&hidden, 4, 1, f) == 1;
    const bool v2 = ok && std::string(magic, 8) == "FKJNNUE2";
    if (v2) ok = std::fread(&h2, 4, 1, f) == 1 && h2 >= 1 && h2 <= 64 && inputs == kZoneInputs && hidden == kHidden;
    ok = ok && (v2 || std::string(magic, 8) == "FKJNNUE1") && (inputs == kBaseInputs || inputs == kKingInputs || inputs == kZoneInputs) && (hidden == kHidden || hidden == kMaxHidden);
    if (ok) {
        m.inputs = int(inputs);
        m.king_relative = inputs == kKingInputs;
        m.king_zone = inputs == kZoneInputs;
        m.hidden = int(hidden);
        m.w.resize(std::size_t(m.inputs) * m.hidden);
        ok = std::fread(m.w.data(), 4, m.w.size(), f) == m.w.size() &&
             std::fread(m.b.data(), 4, hidden, f) == hidden && std::fread(m.v.data(), 4, hidden, f) == hidden &&
             std::fread(&m.o, 4, 1, f) == 1;
        if (ok && v2) {
            m.l2.resize(std::size_t(h2) * 2 * hidden);
            ok = std::fread(m.l2.data(), 4, m.l2.size(), f) == m.l2.size() &&
                 std::fread(m.b2.data(), 4, h2, f) == h2 && std::fread(m.w3.data(), 4, h2, f) == h2;
            m.h2 = int(h2);
        }
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
// H はコンパイル時定数にする(実行時の値にすると内側のループが遅くなる。128で約3〜4割の速度低下を実測)。
// persp(0=先手,1=後手)から見た特徴を足し込む(中間層の和、バイアス前)。
template <int H>
inline void accumulate(const rules::Snapshot& s, const int persp, float* out) {
    const Model& m = model();
    float acc[H] = {};     // 手元の配列に足し込む(引数のポインタへ直接足すと、重みと重なる可能性を疑って最適化が弱くなり約25%遅くなる)
    auto add = [&](int idx) {
        const float* row = &m.w[std::size_t(idx) * H];
        for (int h = 0; h < H; ++h) acc[h] += row[h];
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
        if ((m.king_relative || m.king_zone) && p.kind != 8) {
            const int nk = nnue_kind(p.kind, p.promoted);
            const int kind = nk - 1 - (nk > 8 ? 1 : 0);
            const int direction = persp == 0 ? 1 : -1;
            for (int anchor = 0; anchor < 2; ++anchor) {
                const int king = kings[anchor == 0 ? persp : 1 - persp];
                if (king < 0) continue;
                if (m.king_zone) {   // 玉周辺±2升(5x5)のみ
                    const int dr = direction * (py / 9 - king / 9), dc = direction * (py % 9 - king % 9);
                    if (dr < -2 || dr > 2 || dc < -2 || dc > 2) continue;
                    add(kBaseInputs + (((anchor * 2 + owner) * 13 + kind) * 5 + dr + 2) * 5 + dc + 2);
                    continue;
                }
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
    for (int h = 0; h < H; ++h) out[h] = acc[h];
}
template <int H>
inline double logit_impl(const rules::Snapshot& s) {
    const Model& m = model();
    const int persp = s.turn == rules::Color::Black ? 0 : 1;
    float acc[H] = {};
    accumulate<H>(s, persp, acc);
    float value = m.o;
    for (int h = 0; h < H; ++h) value += std::clamp(acc[h] + m.b[h], 0.f, 1.f) * m.v[h];
    return value;
}
// 差分更新(隠れ層128・玉周辺特徴の正式形のみ)。直前に同じ視点で評価した盤面との差だけを足し引きする。
// 探索の中身には触れず、評価器の中だけで完結する。既定は無効(毎回全計算)。環境変数 FUKAJI_NNUE_INCREMENTAL=1 で有効。
// 玉が動いたとき・差が多いとき・一定回数ごとには全計算し直す(足し引きの丸め誤差をためない)。全計算との差は最大0.003cp程度(手元実測)。
struct IncCache {
    bool valid = false;
    int updates = 0;
    int kings[2] = {-1, -1};
    std::uint8_t code[81] = {};
    std::uint8_t hand[2][7] = {};
    float acc[kHidden] = {};
};
inline bool incremental_enabled() {
    static const bool on = [] { const char* e = std::getenv("FUKAJI_NNUE_INCREMENTAL"); return e && e[0] == '1'; }();
    return on;
}
inline double logit_incremental(const rules::Snapshot& s) {
    constexpr int H = kHidden;
    const Model& m = model();
    const int persp = s.turn == rules::Color::Black ? 0 : 1;
    thread_local IncCache caches[2];
    IncCache& c = caches[persp];
    std::uint8_t code[81];
    int kings[2] = {-1, -1};
    for (int sq = 0; sq < 81; ++sq) {
        const auto& p = s.board[sq];
        code[sq] = p.kind ? std::uint8_t(p.kind | (p.promoted ? 16 : 0) | (p.color == rules::Color::Black ? 0 : 32)) : 0;
        if (p.kind == 8) kings[p.color == rules::Color::Black ? 0 : 1] = (sq % 9) * 9 + (9 - (sq / 9 + 1));
    }
    int diffs[12], nd = 0;
    bool full = !c.valid || c.updates >= 128 || kings[0] != c.kings[0] || kings[1] != c.kings[1];
    if (!full)
        for (int sq = 0; sq < 81; ++sq)
            if (code[sq] != c.code[sq]) { if (nd == 12) { full = true; break; } diffs[nd++] = sq; }
    if (full) {
        accumulate<H>(s, persp, c.acc);
        c.updates = 0; c.valid = true; c.kings[0] = kings[0]; c.kings[1] = kings[1];
        for (int sq = 0; sq < 81; ++sq) c.code[sq] = code[sq];
        for (int color = 0; color < 2; ++color)
            for (int k = 0; k < 7; ++k) c.hand[color][k] = std::uint8_t(std::min(20, s.hands[color][k]));
    } else {
        float* acc = c.acc;
        auto row = [&](int idx, float sign) {
            const float* r = &m.w[std::size_t(idx) * H];
            for (int h = 0; h < H; ++h) acc[h] += sign * r[h];
        };
        // accumulate() と同じ特徴番号(駒1枚ぶん)。玉は絶対位置の行だけ(玉が動いたら全計算に回るのでここには来ない)。
        auto piece = [&](int sq, std::uint8_t cd, float sign) {
            const int kindk = cd & 15, color = (cd & 32) ? 1 : 0; const bool promoted = (cd & 16) != 0;
            const int file = sq / 9 + 1, rank = sq % 9;
            const int py = rank * 9 + (9 - file);
            const int owner = color ^ persp;
            const int relative = persp ? 80 - py : py;
            const int nk = nnue_kind(kindk, promoted);
            row((owner * 14 + nk - 1) * 81 + relative, sign);
            if (kindk == 8) return;
            const int kind = nk - 1 - (nk > 8 ? 1 : 0);
            const int direction = persp == 0 ? 1 : -1;
            for (int anchor = 0; anchor < 2; ++anchor) {
                const int king = kings[anchor == 0 ? persp : 1 - persp];
                if (king < 0) continue;
                const int dr = direction * (py / 9 - king / 9), dc = direction * (py % 9 - king % 9);
                if (dr < -2 || dr > 2 || dc < -2 || dc > 2) continue;
                row(kBaseInputs + (((anchor * 2 + owner) * 13 + kind) * 5 + dr + 2) * 5 + dc + 2, sign);
            }
        };
        for (int i = 0; i < nd; ++i) {
            const int sq = diffs[i];
            if (c.code[sq]) piece(sq, c.code[sq], -1.f);
            if (code[sq]) piece(sq, code[sq], 1.f);
            c.code[sq] = code[sq];
        }
        static constexpr int map_hand[7] = {0, 1, 2, 3, 5, 6, 4};
        for (int color = 0; color < 2; ++color)
            for (int k = 0; k < 7; ++k) {
                const int n = std::min(20, s.hands[color][k]), old = c.hand[color][k];
                const int base = 2268 + ((color ^ persp) * 7 + map_hand[k]) * 20;
                for (int j = old; j < n; ++j) row(base + j, 1.f);
                for (int j = n; j < old; ++j) row(base + j, -1.f);
                c.hand[color][k] = std::uint8_t(n);
            }
        ++c.updates;
    }
    float value = m.o;
    for (int h = 0; h < H; ++h) value += std::clamp(c.acc[h] + m.b[h], 0.f, 1.f) * m.v[h];
    return value;
}
// G2: 両視点＋2層目(隠れ層128のみ)。
inline double logit_v2(const rules::Snapshot& s) {
    constexpr int H = kHidden;
    const Model& m = model();
    const int persp = s.turn == rules::Color::Black ? 0 : 1;
    float c[2 * H] = {};
    accumulate<H>(s, persp, c);
    accumulate<H>(s, 1 - persp, c + H);
    float value = m.o;
    for (int h = 0; h < H; ++h) {
        c[h] = std::clamp(c[h] + m.b[h], 0.f, 1.f);
        c[H + h] = std::clamp(c[H + h] + m.b[h], 0.f, 1.f);
        value += c[h] * m.v[h];
    }
    // 2層目: 入力側から足し込む形(0の入力は飛ばせる。内側は長さh2の単純な足し込みで速い)。
    float p[64];
    for (int j = 0; j < 64; ++j) p[j] = m.b2[j];
    for (int i = 0; i < 2 * H; ++i) {
        const float ci = c[i];
        if (ci == 0.f) continue;
        const float* row = &m.l2[std::size_t(i) * m.h2];
        for (int j = 0; j < m.h2; ++j) p[j] += ci * row[j];
    }
    for (int j = 0; j < m.h2; ++j) value += std::clamp(p[j], 0.f, 1.f) * m.w3[j];
    return value;
}
inline double logit_side_to_move(const rules::Snapshot& s) {
    if (model().h2) return logit_v2(s);
    if (model().king_zone && model().hidden == kHidden && incremental_enabled()) return logit_incremental(s);
    return model().hidden == kHidden ? logit_impl<kHidden>(s) : logit_impl<kMaxHidden>(s);
}
inline int eval_black(const rules::Snapshot& s) {
    const double l = std::clamp(logit_side_to_move(s), -20.0, 20.0);
    const double cp = l * model().scale;
    return int(s.turn == rules::Color::Black ? cp : -cp);
}
} // namespace nnue_hybrid
} // namespace shogi::strategy
