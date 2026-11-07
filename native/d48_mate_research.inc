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
