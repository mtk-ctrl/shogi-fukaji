// FUKAJI research-only independent C++ attack/repetition probe.
// No engine integration. Research-only legal-move generator and reversible state probe.
// Protocol: C <b|w> <SFEN> => whether that king is attacked; R <SFEN>
// Stateful L lists legal moves; T classifies fourfold/continuous-check outcomes.
// => fourfold state; Z => reset repetition history.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
struct Piece { char type=0; bool black=false, promoted=false; };
struct Board {
  array<Piece,81> squares{};
  string turn, hand;
  int kings[2]={-1,-1};
  int ply=1;
};
static Board parse(const string& sfen) {
  istringstream in(sfen);
  string position, turn, hand, ply;
  if (!(in>>position>>turn>>hand>>ply) || (turn!="b"&&turn!="w"))
    throw runtime_error("invalid SFEN fields");
  Board b; b.turn=turn; b.hand=hand;
  if(ply.empty() || !all_of(ply.begin(),ply.end(),[](char c){return c>='0'&&c<='9';}))
    throw runtime_error("invalid move number");
  b.ply=stoi(ply);
  if(b.ply<1)throw runtime_error("invalid move number");
  int row=0,col=0; bool promote=false;
  for(char ch:position) {
    if(ch=='/') { if(col!=9||row>=8)throw runtime_error("rank width"); ++row;col=0;continue; }
    if(ch=='+') {if(promote)throw runtime_error("double promotion");promote=true;continue;}
    if(isdigit(static_cast<unsigned char>(ch))) {
      if(promote||ch<'1'||ch>'9')throw runtime_error("empty squares");
      col+=ch-'0'; if(col>9)throw runtime_error("rank overflow");
      continue;
    }
    char kind=static_cast<char>(toupper(static_cast<unsigned char>(ch)));
    if(string("PLNSGBRK").find(kind)==string::npos||row>8||col>=9)
      throw runtime_error("invalid piece");
    if(promote && string("PLNSBR").find(kind)==string::npos)
      throw runtime_error("invalid promotion");
    bool black=isupper(static_cast<unsigned char>(ch));
    b.squares[row*9+col]={kind,black,promote};
    if(kind=='K') {
      int index=black?0:1;
      if(b.kings[index]>=0)throw runtime_error("duplicate king");
      b.kings[index]=row*9+col;
    }
    ++col;promote=false;
  }
  if(row!=8||col!=9||promote)throw runtime_error("incomplete board");
  return b;
}
static bool clear_path(const Board& b,int r,int c,int tr,int tc) {
  int sr=(tr>r)-(tr<r),sc=(tc>c)-(tc<c);
  r+=sr;c+=sc;
  while(r!=tr||c!=tc) {
    if(b.squares[r*9+c].type)return false;
    r+=sr;c+=sc;
  }
  return true;
}
static bool attacks(const Board& b,int r,int c,int tr,int tc,Piece p) {
  int dr=tr-r,dc=tc-c,dir=p.black?-1:1;
  int ar=abs(dr),ac=abs(dc);
  if(!dr&&!dc)return false;
  if(p.type=='K')return ar<=1&&ac<=1;
  if(p.promoted && string("PLNS").find(p.type)!=string::npos)
    return (dr==dir&&ac<=1)||(dr==0&&ac==1)||(dr==-dir&&dc==0);
  switch(p.type) {
    case 'P':return dr==dir&&dc==0;
    case 'L':return dc==0&&dr*dir>0&&clear_path(b,r,c,tr,tc);
    case 'N':return dr==2*dir&&ac==1;
    case 'S':return (dr==dir&&ac<=1)||(dr==-dir&&ac==1);
    case 'G':return (dr==dir&&ac<=1)||(dr==0&&ac==1)||(dr==-dir&&dc==0);
    case 'B':return (ar==ac&&clear_path(b,r,c,tr,tc))
       ||(p.promoted&&ar+ac==1);
    case 'R':return (((dr==0)!=(dc==0))&&clear_path(b,r,c,tr,tc))
       ||(p.promoted&&ar==1&&ac==1);
  }
  return false;
}
static bool checked(const Board& b,bool black) {
  int king=b.kings[black?0:1];
  if(king<0)throw runtime_error("missing king");
  int tr=king/9,tc=king%9;
  for(int sq=0;sq<81;++sq) {
    Piece p=b.squares[sq];
    if(p.type && p.black!=black && attacks(b,sq/9,sq%9,tr,tc,p))
      return true;
  }
  return false;
}
static string state_key(const Board& b) {
  // Normalized position/turn/hands; ignore the SFEN move number.
  // Caller supplies the same canonical hand ordering as python-shogi.
  string key;
  for(const Piece& p:b.squares) {
    key+=p.type?p.type:'.';
    key+=p.black?'b':'w';
    key+=p.promoted?'+':'-';
  }
  return key+"|"+b.turn+"|"+b.hand;
}

// Stage 3: independently enumerate pseudo-legal moves and full legal moves.
// H additionally forbids pawn-drop mate via independent native reply search.
// Engine integration and speed claims are out of scope.
static bool can_stay(Piece p,int row) {
  if(p.promoted || (p.type!='P'&&p.type!='L'&&p.type!='N'))return true;
  return p.black ? row>=(p.type=='N'?2:1) : row<=(p.type=='N'?6:7);
}
static bool can_promote(Piece p,int from_row,int to_row) {
  if(p.promoted || string("PLNSBR").find(p.type)==string::npos)return false;
  return p.black ? (from_row<=2||to_row<=2) : (from_row>=6||to_row>=6);
}
static string square_usi(int sq) {
  return to_string(9-sq%9)+string(1,static_cast<char>('a'+sq/9));
}
static array<array<int,7>,2> parse_hands(const string& hand) {
  array<array<int,7>,2> counts{};
  if(hand=="-")return counts;
  int number=0;
  const string kinds="PLNSGBR";
  for(char ch:hand) {
    if(isdigit(static_cast<unsigned char>(ch))) {
      number=number*10+(ch-'0');
      if(number>100)throw runtime_error("hand count out of range");
      continue;
    }
    char kind=static_cast<char>(toupper(static_cast<unsigned char>(ch)));
    auto i=kinds.find(kind);
    if(i==string::npos)throw runtime_error("invalid hand piece");
    counts[isupper(static_cast<unsigned char>(ch))?0:1][i]+=(number?number:1);
    number=0;
  }
  if(number)throw runtime_error("dangling hand count");
  return counts;
}
static vector<string> pseudo_moves(const Board& b,bool self_check_filter) {
  vector<string> result;
  bool side=b.turn=="b";
  for(int from=0;from<81;++from) {
    Piece p=b.squares[from];
    if(!p.type||p.black!=side)continue;
    for(int to=0;to<81;++to) {
      Piece target=b.squares[to];
      if(target.type&&target.black==side)continue;
      if(!attacks(b,from/9,from%9,to/9,to%9,p))continue;
      bool plain=can_stay(p,to/9);
      bool promo=can_promote(p,from/9,to/9);
      if(!plain&&!promo)continue;
      if(self_check_filter) {
        Board next=b;
        next.squares[from]={};
        next.squares[to]=p;
        if(p.type=='K')next.kings[side?0:1]=to;
        if(checked(next,side))continue;
      }
      string move=square_usi(from)+square_usi(to);
      if(plain)result.push_back(move);
      if(promo)result.push_back(move+"+");
    }
  }
  const string kinds="PLNSGBR";
  auto hands=parse_hands(b.hand);
  for(size_t i=0;i<kinds.size();++i) {
    if(hands[side?0:1][i]==0)continue;
    Piece p{kinds[i],side,false};
    for(int to=0;to<81;++to) {
      if(b.squares[to].type||!can_stay(p,to/9))continue;
      if(p.type=='P') {
        bool nifu=false;
        for(int r=0;r<9;++r) {
          Piece on_file=b.squares[r*9+to%9];
          if(on_file.type=='P'&&on_file.black==side&&!on_file.promoted) {
            nifu=true;break;
          }
        }
        if(nifu)continue;
      }
      if(self_check_filter) {
        Board next=b;
        next.squares[to]=p;
        if(checked(next,side))continue;
        // A pawn drop is forbidden only if that pawn directly checks the
        // opposing king AND the opponent has no self-check-safe reply.
        // Against adjacent pawn check, a defensive drop cannot interpose,
        // so the recursive reply enumeration cannot recurse into this rule.
        if(p.type=='P') {
          int enemy_king=b.kings[side?1:0];
          if(enemy_king>=0 &&
             attacks(next,to/9,to%9,enemy_king/9,enemy_king%9,p)) {
            next.turn=side?"w":"b";
            if(pseudo_moves(next,true).empty())continue;
          }
        }
      }
      result.push_back(string(1,p.type)+"*"+square_usi(to));
    }
  }
  sort(result.begin(),result.end());
  if(adjacent_find(result.begin(),result.end())!=result.end())
    throw runtime_error("duplicate generated move");
  return result;
}
static string encode_moves(const vector<string>& moves) {
  string output;
  for(const auto& move:moves) {
    if(!output.empty())output+=",";
    output+=move;
  }
  return output;
}


static string encode_hand(const array<array<int,7>,2>& counts) {
  const string order="RBGSNLP", kinds="PLNSGBR";
  string output;
  for(int side=0;side<2;++side)for(char kind:order) {
    int n=counts[side][kinds.find(kind)];
    if(n<0)throw runtime_error("negative hand count");
    if(n==0)continue;
    if(n>1)output+=to_string(n);
    output+=static_cast<char>(side==0?kind:tolower(static_cast<unsigned char>(kind)));
  }
  return output.empty()?"-":output;
}
static string format_sfen(const Board& b) {
  string output;
  for(int row=0;row<9;++row) {
    if(row)output+='/';
    int empty=0;
    for(int col=0;col<9;++col) {
      Piece p=b.squares[row*9+col];
      if(!p.type){++empty;continue;}
      if(empty){output+=to_string(empty);empty=0;}
      if(p.promoted)output+='+';
      output+=static_cast<char>(p.black?p.type:tolower(static_cast<unsigned char>(p.type)));
    }
    if(empty)output+=to_string(empty);
  }
  return output+" "+b.turn+" "+encode_hand(parse_hands(b.hand))+" "+to_string(b.ply);
}
static int from_usi_square(char file,char rank) {
  if(file<'1'||file>'9'||rank<'a'||rank>'i')throw runtime_error("invalid USI square");
  return (rank-'a')*9+('9'-file);
}
static void apply_legal_move(Board& b,const string& move) {
  // Validate with the independent full native legal generator before mutating.
  vector<string> legal=pseudo_moves(b,true);
  if(!binary_search(legal.begin(),legal.end(),move))throw runtime_error("illegal move");
  bool side=b.turn=="b";
  auto hands=parse_hands(b.hand);
  const string kinds="PLNSGBR";
  if(move.size()==4&&move[1]=='*') {
    char kind=move[0];
    size_t index=kinds.find(kind);
    if(index==string::npos||hands[side?0:1][index]<=0)
      throw runtime_error("drop without piece in hand");
    int to=from_usi_square(move[2],move[3]);
    if(b.squares[to].type)throw runtime_error("occupied drop square");
    b.squares[to]={kind,side,false};
    --hands[side?0:1][index];
  } else if(move.size()==4||(move.size()==5&&move[4]=='+')) {
    int from=from_usi_square(move[0],move[1]);
    int to=from_usi_square(move[2],move[3]);
    Piece moving=b.squares[from], captured=b.squares[to];
    if(!moving.type||moving.black!=side)throw runtime_error("invalid moving piece");
    if(captured.type) {
      size_t index=kinds.find(captured.type);
      if(index==string::npos)throw runtime_error("cannot capture a king");
      ++hands[side?0:1][index];
    }
    b.squares[from]={};
    if(move.size()==5)moving.promoted=true;
    b.squares[to]=moving;
    if(moving.type=='K')b.kings[side?0:1]=to;
  } else throw runtime_error("invalid move format");
  b.hand=encode_hand(hands);
  b.turn=side?"w":"b";
  ++b.ply;
}

// Distinguish ordinary fourfold repetition from a continuous-check loss.
// Each move record describes the mover and whether the opponent was checked
// immediately after that move. Examine all moves between the first and fourth
// occurrence of the most recent repeated position (inclusive interval).
static string repetition_verdict(const vector<string>& history,
                                 const vector<bool>& mover_black,
                                 const vector<bool>& gave_check) {
  if(history.empty())return "none";
  vector<size_t> matches;
  for(size_t i=0;i<history.size();++i)
    if(history[i]==history.back())matches.push_back(i);
  if(matches.size()<4)return "none";
  if(mover_black.size()+1!=history.size() || gave_check.size()!=mover_black.size())
    throw runtime_error("inconsistent repetition history");
  size_t start=matches[matches.size()-4], end=matches.back();
  bool all_black=true, all_white=true;
  int black_moves=0,white_moves=0;
  for(size_t i=start;i<end;++i) {
    if(mover_black[i]) {++black_moves;if(!gave_check[i])all_black=false;}
    else {++white_moves;if(!gave_check[i])all_white=false;}
  }
  bool black_check=black_moves>0&&all_black;
  bool white_check=white_moves>0&&all_white;
  if(black_check&&!white_check)return "black_loses";
  if(white_check&&!black_check)return "white_loses";
  return "draw";
}

int main() {
  unordered_map<string,int> counts;
  optional<Board> current;
  vector<Board> undo;
  vector<string> history;
  vector<bool> mover_black, gave_check;
  string line;
  while(getline(cin,line)) {
    if(line=="Z"){counts.clear();current.reset();undo.clear();history.clear();mover_black.clear();gave_check.clear();cout<<"OK\n";continue;}
    try {
      if(line=="S"||line=="F"||line=="U"||line=="L"||line=="T") {
        if(!current)throw runtime_error("no loaded board");
        if(line=="S")cout<<format_sfen(*current)<<"\n";
        else if(line=="L")cout<<encode_moves(pseudo_moves(*current,true))<<"\n";
        else if(line=="T")cout<<repetition_verdict(history,mover_black,gave_check)<<"\n";
        else if(line=="F") {
          int repeats=0;
          for(const auto& key:history)if(key==history.back())++repeats;
          cout<<(repeats>=4?1:0)<<"\n";
        } else {
          if(undo.empty())throw runtime_error("undo stack empty");
          *current=undo.back();undo.pop_back();history.pop_back();mover_black.pop_back();gave_check.pop_back();
          cout<<format_sfen(*current)<<"\n";
        }
        continue;
      }
      if(line.size()<3||line[1]!=' ')throw runtime_error("protocol");
      if(line[0]=='B') {
        Board b=parse(line.substr(2));
        (void)parse_hands(b.hand);
        b.hand=encode_hand(parse_hands(b.hand));
        current=b;undo.clear();history={state_key(b)};
        mover_black.clear();gave_check.clear();
        cout<<format_sfen(b)<<"\n";
        continue;
      }
      if(line[0]=='M') {
        if(!current)throw runtime_error("no loaded board");
        Board next=*current;
        bool moved_black=next.turn=="b";
        apply_legal_move(next,line.substr(2));
        bool delivered_check=checked(next,next.turn=="b");
        undo.push_back(*current);*current=next;
        history.push_back(state_key(next));
        mover_black.push_back(moved_black);gave_check.push_back(delivered_check);
        cout<<format_sfen(next)<<"\n";
        continue;
      }
      if(line[0]=='Q') {
        // Bounded research-only core timing: exclude SFEN parsing and all I/O.
        // Q <repetitions> <SFEN> -> total_legal_moves elapsed_nanoseconds
        istringstream args(line.substr(2));
        int repetitions=0;
        if(!(args>>repetitions)||repetitions<1||repetitions>5000)
          throw runtime_error("invalid benchmark repetitions");
        string sfen;getline(args,sfen);
        Board b=parse(sfen);
        (void)parse_hands(b.hand);
        volatile uint64_t count=0;
        auto started=chrono::steady_clock::now();
        for(int i=0;i<repetitions;++i)count+=pseudo_moves(b,true).size();
        auto elapsed=chrono::duration_cast<chrono::nanoseconds>(
          chrono::steady_clock::now()-started).count();
        cout<<count<<" "<<elapsed<<"\n";
        continue;
      }
      if(line[0]=='C') {
        if(line.size()<6||line[3]!=' '||(line[2]!='b'&&line[2]!='w'))
          throw runtime_error("check protocol");
        cout<<(checked(parse(line.substr(4)),line[2]=='b')?1:0)<<"\n";
      } else if(line[0]=='P'||line[0]=='H') {
        Board b=parse(line.substr(2));
        cout<<encode_moves(pseudo_moves(b,line[0]=='H'))<<"\n";
      } else if(line[0]=='R') {
        string key=state_key(parse(line.substr(2)));
        cout<<(++counts[key]>=4?1:0)<<"\n";
      } else throw runtime_error("unknown operation");
    } catch(const exception& e) {
      cerr<<"native rules probe rejected input: "<<e.what()<<"\n";
      return 2;
    }
  }
}