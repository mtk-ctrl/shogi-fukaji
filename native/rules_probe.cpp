// FUKAJI research-only independent C++ attack/repetition probe.
// No engine integration. This is NOT a complete native legal-move generator.
// Protocol: C <b|w> <SFEN> => whether that king is attacked; R <SFEN>
// => fourfold state; Z => reset repetition history.
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <iostream>
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
};
static Board parse(const string& sfen) {
  istringstream in(sfen);
  string position, turn, hand, ply;
  if (!(in>>position>>turn>>hand>>ply) || (turn!="b"&&turn!="w"))
    throw runtime_error("invalid SFEN fields");
  Board b; b.turn=turn; b.hand=hand;
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

// Stage 2: independently enumerate pseudo-legal moves and self-check-filtered
// moves. H intentionally DOES NOT implement the pawn-drop-mate prohibition.
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

int main() {
  unordered_map<string,int> counts;
  string line;
  while(getline(cin,line)) {
    if(line=="Z"){counts.clear();cout<<"OK\n";continue;}
    try {
      if(line.size()<3||line[1]!=' ')throw runtime_error("protocol");
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