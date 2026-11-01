// Independent research backend. Includes the frozen reference without editing it.
#include "nnue_search_ffi.cpp"

namespace {
struct NAccumulatorSearch : NSearch {
  array<array<double,kNHidden>,2> acc{};
  uint64_t feature_updates=0;
  NAccumulatorSearch(const NModel& m,const NativeState& s):NSearch(m,s) {
    for(int sq=0;sq<81;++sq)piece(board.squares[sq],sq,1);
    const auto hands=parse_hands(board.hand);
    for(int side=0;side<2;++side)for(int kind=0;kind<7;++kind)
      for(int j=0;j<min(20,hands[side][kind]);++j)hand(side,kind,j,1);
  }
  void feature(int perspective,size_t index,int sign) {
    if(index>=kNInputs)throw runtime_error("accumulator feature range");
    for(size_t h=0;h<kNHidden;++h)acc[perspective][h]+=sign*static_cast<double>(model.w[index*kNHidden+h]);
    ++feature_updates;
  }
  void piece(const Piece& p,int sq,int sign) {
    if(!p.type)return;
    for(int perspective=0;perspective<2;++perspective) {
      const int owner=(p.black?0:1)^perspective,relative=perspective?80-sq:sq;
      feature(perspective,static_cast<size_t>((owner*14+nnue_kind(p)-1)*81+relative),sign);
    }
  }
  void hand(int side,int kind,int j,int sign) {
    for(int perspective=0;perspective<2;++perspective)
      feature(perspective,static_cast<size_t>(2268+((side^perspective)*7+kind)*20+j),sign);
  }
  double value(int perspective) const {
    double result=0.;
    for(size_t h=0;h<kNHidden;++h)result+=clamp(acc[perspective][h]+model.b[h],0.,1.)*model.v[h];
    result+=model.o;
    if(!isfinite(result))throw runtime_error("nonfinite accumulator output");
    return result;
  }
  struct Guard {
    NAccumulatorSearch& s;
    array<array<double,kNHidden>,2> saved;
    NSearch::MoveGuard board_guard;
    Guard(NAccumulatorSearch& search,const string& word):s(search),saved(search.acc),board_guard(search,word) {
      try {
        const int to=from_usi_square(word[2],word[3]);
        if(word[1]!='*') {
          const int from=from_usi_square(word[0],word[1]);
          s.piece(board_guard.saved.squares[from],from,-1);
          s.piece(s.board.squares[from],from,1);
        }
        s.piece(board_guard.saved.squares[to],to,-1);
        s.piece(s.board.squares[to],to,1);
        if(word[1]=='*'||board_guard.saved.squares[to].type) {
          const auto before=parse_hands(board_guard.saved.hand),after=parse_hands(s.board.hand);
          for(int side=0;side<2;++side)for(int kind=0;kind<7;++kind) {
            const int a=min(20,before[side][kind]),b=min(20,after[side][kind]);
            for(int j=a;j<b;++j)s.hand(side,kind,j,1);
            for(int j=b;j<a;++j)s.hand(side,kind,j,-1);
          }
        }
      }catch(...){s.acc=saved;throw;}
    }
    ~Guard() noexcept {s.acc=saved;}
  };
  double search(int depth,double alpha,double beta,int ply) {
    ++nodes;time_check();
    const auto legal=depth==0?pseudo_moves(board,true,true):moves();
    const auto end=terminal(legal);
    if(!end.first.empty())return end.second<0?0.:(kNMate-ply)*(end.second==(board.turn=="b"?0:1)?1.:-1.);
    if(depth==0){++evals;return clamp(value(board.turn=="b"?0:1),-20.,20.)*1000.;}
    double best=-2*kNMate;string word;
    for(const auto& move:legal) {
      double score;
      {Guard guard(*this,move);score=-search(depth-1,-beta,-alpha,ply+1);}
      if(score>best){best=score;word=move;}alpha=max(alpha,best);if(alpha>=beta)break;
    }
    if(!word.empty())hints[history.back()]=word;
    return best;
  }
};
}

extern "C" int fukaji_nnue_accumulator_value(uint64_t model,uint64_t state,int perspective,double* output) noexcept {
  if(!output||perspective<0||perspective>1)return -1;
  try {
    lock_guard<mutex> lock(state_mutex);
    NAccumulatorSearch search(nnue_models.at(model),find_state(state));
    *output=search.value(perspective);return 0;
  }catch(...){return -3;}
}

extern "C" int fukaji_nnue_accumulator_search(uint64_t model,uint64_t state,int milliseconds,int max_depth,
                                            int fixed_depth,char* output,size_t capacity) noexcept {
  if(milliseconds<1||milliseconds>10000||max_depth<1||max_depth>64||(fixed_depth!=0&&fixed_depth!=1))return -1;
  try {
    lock_guard<mutex> lock(state_mutex);
    const auto started=chrono::steady_clock::now();
    NAccumulatorSearch search(nnue_models.at(model),find_state(state));
    const auto original_acc=search.acc;
    search.deadline=fixed_depth?chrono::steady_clock::time_point::max():started+chrono::milliseconds(milliseconds);
    auto moves=search.moves(false);const auto end=search.terminal(moves);
    string best=moves.empty()||!end.first.empty()?"":moves[0];
    int completed=0,partial=0;double score=0.;bool has_score=false;
    for(int depth=1;depth<=max_depth&&!best.empty();++depth) {
      string current=best;double value=-2*kNMate,alpha=-2*kNMate;int searched=0;
      try {
        for(const auto& move:moves) {
          search.time_check();double v;
          {NAccumulatorSearch::Guard guard(search,move);v=-search.search(depth-1,-2*kNMate,-alpha,1);}
          if(v>value){value=v;current=move;}alpha=max(alpha,value);++searched;
        }
      }catch(const NTimeout&){if(completed==0&&searched){best=current;score=value;has_score=true;partial=searched;}break;}
      best=current;score=value;has_score=true;completed=depth;
      stable_sort(moves.begin(),moves.end(),[&](const string& a,const string& b){return (a==best)>(b==best);});
      if(abs(score)>kNMate-1000.)break;
    }
    if(state_key(search.board)!=find_state(state).history.back()||search.history!=find_state(state).history||search.acc!=original_acc)
      throw runtime_error("accumulator/board/history restoration failed");
    const double elapsed=chrono::duration<double,milli>(chrono::steady_clock::now()-started).count();
    ostringstream out;out<<setprecision(17)<<"{\"move\":";
    if(best.empty())out<<"null";else out<<'"'<<best<<'"';
    out<<",\"depth\":"<<completed<<",\"partial_root_moves\":"<<partial<<",\"root_moves\":"<<moves.size()
       <<",\"nodes\":"<<search.nodes<<",\"evaluations\":"<<search.evals<<",\"score\":";
    if(has_score)out<<score;else out<<"null";
    out<<",\"elapsed_ms\":"<<elapsed<<",\"quiescence_depth\":0,\"quiescence_nodes\":0"
       <<",\"full_accumulator_builds\":2,\"accumulator_feature_updates\":"<<search.feature_updates<<",\"terminal\":";
    if(end.first.empty())out<<"null";else {out<<"{\"status\":\""<<end.first<<"\",\"winner\":";
      if(end.second<0)out<<"null";else out<<end.second;out<<'}';}
    out<<'}';return copy_text(out.str(),output,capacity);
  }catch(...){return -3;}
}
