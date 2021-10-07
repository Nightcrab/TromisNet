#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <stdio.h>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <functional>
#include <queue>
#include <ios>

#define INF32 100000000
#define all(x) x.begin(), x.end()
/*

    TriTris

    A variant of Tetris where each piece is made of 3 tiles instead of 4.

    The size of a board is also now 5x10 instead of 10x20.

    there is a solid wall surrounding the entire matrix so the actual size is 7x12.

    There are no wallkicks.

    All garbage is messy.

    There are no invisible rows. If a filled tile is pushed out of the board it's considered a top-out.

    g++ -pg -g -o tritris.exe tritris.cpp

*/

int ipow(int base, int exp) {
    int result = 1;
    for (;;) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}

int lcg(int *x) {
    long long int k = (22695477*1LL*(*x) + 1) % (1 << 31);
    *x = (int) k;
    return *x;
}

void shuffle (int *bag, int *seed) {
    for (int i=3;i>0;i--)
    {
        int j = lcg(seed) % (i + 1);
        int t = bag[i];
        bag[i] = bag[j];
        bag[j] = t;
    }
}


bool comp (std::vector<int> a, std::vector<int> b) {
    return a[3] > b[3];
}

struct piece {
    std::vector<std::pair<int,int>> tiles;
};

piece pieces[2][4];

struct ArrayHasher {
    std::size_t operator()(const std::array<int, 3>& a) const {
        std::size_t h = 0;

        for (auto e : a) {
            h ^= std::hash<int>{}(e)  + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }   
};

struct board {
    int w = 7;
    int h = 12;
    int tiles[84] = {0};
    int overlay[84] = {0};
    int garbage = 0;
    int attack = 0;
    bool alive = true;
    int bag[8] = {0};
    int bagPos = 0;
    int seed[2];
    int combo = 0;
    int piece_count = 0;
    int pos[3] = {0};
    int type = 0;
    int htype = 0;
    int clear_size = 0;
    bool graphical = true;
    std::unordered_map<std::array<int, 3>, std::array<int, 4>, ArrayHasher> spt;

    int height (int k) {
        int h = 0;
        for (int i=k+7;i<84;i+=7) {
            if (tiles[i] != 0) {
                break;
            }
            else {
                h++;
            }
        }
        return h;
    }
    int row (int k) {
        return (k-(k % w))/w;
    }
    int dist (int k, int dir) {
        int d = 0;
        for (int i=k+dir;i<(row(k)+1)*w && i>row(k)*w;i+=dir) {
            if (tiles[i] != 0) {
                break;
            }
            else {
                d++;
            }
        }
        return d;
    }
    int filled(int x, int y) {
        return (tiles[w*y+x] != 0);
    }
    void fill(int x, int y, int val) {
        tiles[w*y+x] = val;
    }
    void empty(int x, int y) {
        tiles[w*y+x] = 0;
    }
    void clearLine(int y) {
        for (int i=(y+1)*w-1;i>=w;i--) {
            tiles[i] = tiles[i-w];
        }
        tiles[0] = 8;
        tiles[w] = 8;
        for (int i=1;i<w-1;i++) {
            tiles[i] = 0;
        }
    }
    void addGarbage() {
        lcg(&seed[0]);
        int hole = seed[0] % (w-2) + 1;

        for (int i=1;i<w-1;i++) {
            if (tiles[i] != 0) {
                alive = false;
            }
        }
        for (int i=0;i<w*(h-1);i++) {
            tiles[i] = tiles[i+w];
        }
        for (int i=w*(h-1);i<w*h;i++) {
            tiles[i] = 8;
        }

        tiles[w*(h-2)+hole] = 0;
    }
    void emptyMeter () {
        for (int i=0;i<garbage;i++) {
            addGarbage();
        }
        garbage = 0;
    }
    void nextBag() {
        lcg(&seed[1]);
        int new_bag[4] = {0,0,1,1};
        shuffle(new_bag, &seed[1]);
        for (int i=0;i<4;i++) {
            bag[i] = bag[i+4];
            bag[i+4] = new_bag[i];
        }
    }
    void nextPiece() {
        type = bag[bagPos];
        if (type == 8) {
            for (int i=0;i<8;i++) {
                std::cout << bag[i] << " " << std::endl;
            }
        }
        bagPos++;
        if (bagPos == 4) {
            bagPos = 0;
            nextBag();
        }
        resetPiece();
        if (collides(type,pos[0],pos[1],pos[2])) {
            alive = false;
        }
    }
    std::pair<int,int> pieceTile(int i) {
        piece ghost = pieces[type][pos[2]];
        return std::make_pair(pos[0]+ghost.tiles[i].first, pos[1]+ghost.tiles[i].second);
    }
    void clearLines() {
        for (int i=0;i<h-1;i++) {
            bool filled = true;
            for (int j=1;j<w-1;j++) {
                if (tiles[i*w+j] == 0) {
                    filled = false;
                    break;
                }
            }
            if (filled) {
                clearLine(i);
                clear_size++;
                clearLines();
                return;
            }
        }
    }
    void placePiece() {
        bool on_ground = false;
        for (int i=0;i<3;i++) {
            std::pair<int,int> tile = pieceTile(i);
            if (filled(tile.first, tile.second+1)) {
                on_ground = true;
                break;
            }
        }
        if (!on_ground) {
            return;
        }
        for (int i=0;i<3;i++) {
            std::pair<int,int> tile = pieceTile(i);
            tiles[tile.first+w*tile.second] = type+1;
        }
        clear_size = 0;
        clearLines();
        if (clear_size > 0) {
            combo++;
        }
        if (clear_size == 0) {
            combo = 0;
            emptyMeter();
        }
        attack = clear_size;
        int cancel = std::min(garbage, attack);
        garbage -= cancel;
        attack -= cancel;
        clear_size = 0;
        nextPiece();
        if (bagPos == 4) {
            bagPos = 0;
            nextBag();
        }
        piece_count++;
    }
    void printPiece() {
        if (!graphical) {
            return;
        }
        for (int i=0;i<84;i++) {
            overlay[i] = 0;
        }
        piece active = pieces[type][pos[2]];
        for (int i=0;i<3;i++) {
            int x,y;
            x = pos[0]+active.tiles[i].first;
            y = pos[1]+active.tiles[i].second;
            overlay[y*w + x] = type+1;
            if (tiles[y*w + x] != 0) {
                alive = false;
            }
        }
    }
    void clearAll () {
        for (int i=0;i<84;i++) {
            tiles[i] = 0;
            if (i % w == 0 || i % w == (w-1) || row(i) == h-1) {
                tiles[i] = 8;
            }
        }
    }
    void init () {
        clearAll();
        nextBag();
        nextBag();
        nextPiece();
        resetPiece();
        printPiece();
    }
    void resetPiece() {
        pos[0] = 2;
        pos[1] = 0;
        pos[2] = 0; 
        printPiece();
    }
    void rotate (int theta) {
        int r = (pos[2] + theta) % 4;
        if (!collides(type, pos[0],pos[1],r)) {
            pos[2] = r;
        }
        printPiece();
    }
    void tap (int d) {
        int mn = INF32;
        for (int i=0;i<3;i++) {
            std::pair<int,int> tile = pieceTile(i);
            mn = std::min(mn, dist(tile.first+tile.second*w, d));
        }
        if (mn == 0) {
            return;
        }
        pos[0] += d;
        printPiece();
    }
    void forfeit () {
        alive = false;
    }

    void drop () {
        int h = INF32;
        for (int i=0;i<3;i++) {
            std::pair<int,int> tile = pieceTile(i);
            h = std::min(h,height(tile.first+w*tile.second));
        } 
        pos[1] += h;
        printPiece();
    }
    void harddrop () {
        if (!alive) {
            return;
        }
        drop();
        placePiece();
    }
    bool collides (int t, int x, int y, int r) {
        piece ghost = pieces[t][r];
        for (int i=0;i<3;i++) {
            if (filled(x+ghost.tiles[i].first, y+ghost.tiles[i].second)) {
                return true;
            }
        }
        return false;
    }

    int minheight (int t, int x, int y, int r) {
        piece ghost = pieces[t][r];
        int mn = INF32;
        for (int i=0;i<3;i++) {
            mn = std::min(mn, height(x+ghost.tiles[i].first + w*(y+ghost.tiles[i].second)));
        }
        return mn;
    }

    bool openPos (int t, int x, int y, int r) {
        //cout << x << "," << y << "," << r << " already seen: " << (spt.find({x,y,r}) != spt.end()) << endl;
        return !collides(t,x,y,r) && (spt.find({x,y,r}) == spt.end());
    }

    std::vector<std::vector<int>> pathfind (int t) { //update shortest path tree
        // 0 start. 1,2 left right. 3,4 ccw cw. 5, instant drop
        spt.clear();
        std::vector<std::vector<int>> grounded;

        std::queue<std::vector<int>> queue;

        queue.push({2,0,0,0});
        spt.insert({{2,0,0}, {2,0,0,0}});

 
        while (queue.size() > 0) {
            int x = queue.front()[0];
            int y = queue.front()[1];
            int r = queue.front()[2];
            int d = queue.front()[3];

            if (collides(t,x,y+1,r)) {
                grounded.push_back({x,y,r});
            }

            queue.pop();

            int mh = minheight(t,x,y,r);
            if (openPos(t,x,y,(r+1)%4)) {
                queue.push({x,y,(r+1)%4,d+1});
                spt.insert({{x,y,(r+1)%4},{x,y,r,4}});
            }
            if (openPos(t,x,y,(r+3)%4)) {
                queue.push({x,y,(r+3)%4,d+1});
                spt.insert({{x,y,(r+3)%4},{x,y,r,3}});
            }
            if (openPos(t,x,y+mh,r)) {
                queue.push({x,y+mh,r,d+1});
                spt.insert({{x,y+mh,r},{x,y,r,5}});
            }
            if (openPos(t,x+1,y,r)) {
                queue.push({x+1,y,r,d+1});
                spt.insert({{x+1,y,r},{x,y,r,2}});
            }
            if (openPos(t,x-1,y,r)) {
                queue.push({x-1,y,r,d+1});
                spt.insert({{x-1,y,r},{x,y,r,1}});
            }
        }
        return grounded;
    }
};


struct game {
    board boards[2];
    int h = 12;
    int w = 7;

    int selected = 0;

    void initBoards() {
        boards[0].h = h;
        boards[0].w = w;
        boards[1].h = h;
        boards[1].w = w;
        boards[0].seed[0] = std::rand();
        boards[0].seed[1] = std::rand();
        boards[1].seed[0] = std::rand();
        boards[1].seed[1] = std::rand();
        boards[0].init();
        boards[1].init();
    }

    void tick () {
        boards[1].garbage += boards[0].attack;
        boards[0].attack = 0;
        boards[0].garbage += boards[1].attack;
        boards[1].attack = 0;
    }

    std::string display() {
        std::string out;
        for (int i=0;i<h;i++) {
            for (int j=0;j<w;j++) {
                if (boards[0].overlay[w*i+j] != 0) {
                    out += std::to_string(boards[0].overlay[w*i+j]);
                    out += ' '; continue;
                }
                if (boards[0].tiles[w*i+j] == 0) {
                    out += "  "; continue;
                }
                out += std::to_string(boards[0].tiles[w*i+j]);
                out += ' ';
            }
            out += ' ';
            for (int j=0;j<w;j++) {
                if (boards[1].overlay[w*i+j] != 0) {
                    out += std::to_string(boards[1].overlay[w*i+j]);
                    out += ' '; continue;
                }
                if (boards[1].tiles[w*i+j] == 0) {
                    out += "  "; continue;
                }
                out += std::to_string(boards[1].tiles[w*i+j]);
                out += ' ';
            }
            out += '\n';
        }
        out += '\n';
        for (int i=boards[0].bagPos;i<boards[0].bagPos+4;i++) {
            out += std::to_string(boards[0].bag[i]);
        }
        out += "    ";
        for (int i=boards[1].bagPos;i<boards[1].bagPos+4;i++) {
            out += std::to_string(boards[1].bag[i]);
        }
        out += "\npieces: ";
        out += std::to_string(boards[0].piece_count);
        //out += " garbage: ";
        //out += std::to_string(boards[0].garbage);
        //out += " attack: ";
        //out += std::to_string(boards[0].attack);
        out += "\npieces: ";
        out += std::to_string(boards[1].piece_count);
        //out += " garbage: ";
        //out += std::to_string(boards[1].garbage);
        //out += " attack: ";
        //out += std::to_string(boards[1].attack);
        return out;
    }

    void getMoves () {
        std::vector<std::vector<int>> moves = boards[selected].pathfind(boards[selected].type);
        for (int i=0;i<moves.size();i++) {
            boards[selected].pos[0] = moves[i][0];
            boards[selected].pos[1] = moves[i][1];
            boards[selected].pos[2] = moves[i][2];
            boards[selected].printPiece();
            std::cout << display() << std::endl;
        }
        for (int i=0;i<moves.size();i++) {
            std::cout << moves[i][0] << "," << moves[i][1] << "," << moves[i][2] << std::endl; 
        }
    }

    std::string shortestPaths () {
        board &B = boards[selected];
        std::vector<std::vector<int>> moves = boards[selected].pathfind(B.type);
        std::string out;
        for (int i=0;i<moves.size();i++) {
            out += std::to_string(moves[i][0]) + ":" + std::to_string(moves[i][1]) + ":" + std::to_string(moves[i][2]) + ":";
            std::array<int,3> cur;
            cur[0] = moves[i][0];
            cur[1] = moves[i][1];
            cur[2] = moves[i][2];
            std::string path;
            while (B.spt.at(cur)[3] != 0) {
                path += std::to_string(B.spt.at(cur)[3]);
                int a = B.spt.at(cur)[0];
                int b = B.spt.at(cur)[1];
                int c = B.spt.at(cur)[2];
                cur[0] = a;
                cur[1] = b;
                cur[2] = c;
            }
            std::reverse(all(path));
            out += path + "|";
        }
        return out;
    }

    std::string gameState() {
        if (!boards[0].alive || !boards[1].alive) {
            return "game over";
        }
        std::string out;
        for (int i=selected;i<selected+2;i++) {
            board &B = boards[i % 2];
            for (int j=0;j<84;j++) {
                out += std::to_string(B.tiles[j]);
            }
            out += "|";
            for (int j=B.bagPos;j<B.bagPos+4;j++) {
                out += std::to_string(B.bag[j]);
            }
            out += "|";
            out += std::to_string(B.type);
            out += "|";
            out += std::to_string(B.htype);
            out += "|";
            out += std::to_string(B.alive);
            out += "|";
            out += std::to_string(B.garbage);
            out += "|";
            out += std::to_string(B.combo);
            out += "&";
        }
        out += std::to_string(selected);
        out += "|";
        return out;
    }

    void swap() {
        selected = !selected;
    }
};

void readPieceTable () {
    std::ifstream in("piece_table.txt");
    for (int i=0;i<2;i++) {
        int type;
        in >> type;
        for (int r=0;r<4;r++) {  
            piece p;   
            for (int y=0;y<3;y++) {
                for (int x=0;x<3;x++) {
                    char k;
                    in >> k;
                    if (k != '0') {
                        p.tiles.push_back({x,y});
                    }
                }
            }
            pieces[type][r] = p;
        }

    }
}

int main (int argc, char **argv) {  

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    readPieceTable();

    std::ofstream outf("replays.txt");
    //char rf[13];
    //std::strcpy(rf,"replays");
    //std::strcat(rf, argv[1]);
    //std::strcat(rf,".txt");
    //int save = ((int) argv[2][0]) - 48;

    std::vector<game> games;

    int batch_size; 
    std::sscanf(argv[1], "%d", &batch_size);

    std::srand(time(0));

    for (int i=0;i<batch_size;i++) {
        game g;
        g.initBoards();
        games.push_back(g);
        std::cout << games[i].gameState() << "\n";
        std::cout << games[i].shortestPaths() << "\n";
        games[i].swap();
        std::cout << games[i].shortestPaths() << "\n";
        games[i].swap();

    }

    std::cout << std::flush;

    while (true) {
        std::string out;
        for (int i=0;i<batch_size;i++) {
            for (int p=0;p<2;p++) {
                std::vector<int> moves;
                while (true) {
                    int k;
                    std::cin >> k;
                    if (k == 6) {
                        break;
                    }
                    moves.push_back(k);
                }
                for (int j=0;j<moves.size();j++) {
                    int move = moves[j];
                    if (move == 0) {
                        games[i].boards[games[i].selected].forfeit();
                    }
                    if (move == 1) {
                        games[i].boards[games[i].selected].tap(-1);
                    }
                    if (move == 2) {
                        games[i].boards[games[i].selected].tap(1);
                    }
                    if (move == 3) {
                        games[i].boards[games[i].selected].rotate(3);
                    }
                    if (move == 4) {
                        games[i].boards[games[i].selected].rotate(1);
                    }
                    if (move == 5) {
                        games[i].boards[games[i].selected].drop();
                    }
                }

                games[i].boards[games[i].selected].harddrop();
                games[i].tick();
                games[i].swap();
            }

            if (!games[i].boards[0].alive && !games[i].boards[1].alive) {
                out += "draw\n";
                outf << "draw\n";
            }
            else if (!games[i].boards[0].alive) {
                out += "loss\n";
                outf << "loss\n";
            }
            else if (!games[i].boards[1].alive) {
                out += "win\n";
                outf << "win\n";
            }

            if (!games[i].boards[0].alive || !games[i].boards[1].alive) {
                game g;
                g.initBoards();
                games[i] = g;
            }

            out += games[i].gameState() + "\n";
            outf << games[i].display() << "\n";
            out += games[i].shortestPaths() + "\n";
            games[i].swap();
            out += games[i].shortestPaths() + "\n";
            games[i].swap();

        } 
        std::cout << out << std::flush;
    }
    outf.close();
    return 0;
}
