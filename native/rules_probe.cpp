// FUKAJI research-only independent C++ attack/repetition probe.
// No engine integration. This is NOT a complete native legal-move generator.
// Protocol: C <b|w> <SFEN> => whether that king is attacked; R <SFEN>
// => fourfold state; Z => reset repetition history.
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