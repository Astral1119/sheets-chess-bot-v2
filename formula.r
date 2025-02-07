=ArrayFormula(
    let(
        fen,split(D11," "),
        whitesturn,index(fen,2)="w",
        castling,index(fen,3),
        enpassant,index(fen,4),
        ep,if(enpassant="-",9^9,code(enpassant)-96+8*(8-right(enpassant))),
        halfmoves,index(fen,5),
        timer,index(fen,6),
        flip,lambda(in,regexreplace(in,"(.{8})(.{8})(.{8})(.{8})(.{8})(.{8})(.{8})(.{8})","$8$7$6$5$4$3$2$1")),
        colorswap,lambda(in,let(splitup,mid(in,sequence(64),1),join(,if(exact(splitup,upper(splitup)),lower(splitup),upper(splitup))))),
        preboard,substitute(reduce(single(fen),sequence(8),lambda(a,b,substitute(a,b,rept(" ",b)))),"/",),
        board,if(whitesturn,preboard,colorswap(flip(preboard))),
        padding,"-------",
        toppad,padding&padding&padding&"-",
        boardtostring,lambda(in,toppad&join(,padding&in&padding)&toppad),
        iswhite,lambda(in,switch(in,"#","#","-","-"," "," ",exact(in,upper(in)))),
        getactive,lambda(in,(int(find("X",in)/22)-1)*8+mod(find("X",in),22)-7),
        parse,lambda(pos,in,boardtostring(mid(if(ep="-",replace(in,pos,1,"X"),replace(replace(in,pos,1,"X"),ep,1,"#")),sequence(8,1,1,8),8))),

        slide,lambda(in,dist,tdist,
            let(str,index(regexextract(in,{"X((.{"&dist&"}[ #])*(.{"&dist&"}[\w-]))";"([\w-](.{"&dist&"}[ #])*(.{"&dist&"}))X"}),,1),
                con,{iswhite(right(single(str)));iswhite(left(index(str,2)))},
                n,len(str)/(dist+1)-if(con="-",1,con=true),
                vstack(sequence(1,single(n),getactive(in)+tdist,tdist),sequence(1,index(n,2),getactive(in)-tdist,-tdist))
            )),
        step,lambda(in,dist,pcap,tdist,
            let(str,index(regexextract(in,vstack("([\w #].{"&dist&"})X",if(pcap,tocol(,1),"X(.{"&dist&"}[\w #])"))),,1),
                con,vstack(iswhite(left(single(str))),if(pcap,tocol(,1),iswhite(right(index(str,2))))),
                n,if((con="#")+(false=con),1,if(true=con,,if(con="-",NA(),if(pcap,con<>" ",con=" ")))),
                getactive(in)+if(n,,NA())+if(pcap,-tdist,{-tdist;tdist})
            )),
        pawnforward,lambda(in,down,
            let(pos,getactive(in),
                one,regexmatch(in,if(down,"X.{21} "," .{21}X")),
                two,and(regexmatch(in,if(down,"X.{21} .{21} "," .{21} .{21}X")),ceiling(pos/8)=7),
                if(one,if(down,sequence(1,one+two,pos+8,8),sequence(1,one+two,pos-8,-8)),NA())
            )),
        
        bishop,lambda(pos,in,let(parsed,parse(pos,in),vstack(slide(parsed,22,9),slide(parsed,20,7)))),
        rook,lambda(pos,in,let(parsed,parse(pos,in),vstack(slide(parsed,21,8),slide(parsed,0,1)))),
        knight,lambda(pos,in,let(parsed,parse(pos,in),vstack(step(parsed,23,,10),step(parsed,19,,6),step(parsed,44,,17),step(parsed,42,,15)))),
        pawn,lambda(pos,in,let(parsed,parse(pos,in),vstack(pawnforward(parsed,),step(parsed,20,1,7),step(parsed,22,1,9)))),
        king,lambda(pos,in,let(parsed,parse(pos,in),vstack(step(parsed,20,,7),step(parsed,21,,8),step(parsed,22,,9),step(parsed,0,,1)))),
        queen,lambda(pos,in,let(parsed,parse(pos,in),vstack(bishop(pos,in),rook(pos,in)))),

        isattacked,lambda(pos,in,
            or(iferror(vstack(
                exact(mid(in,tocol(bishop(pos,in),3),1),{"b","q"}),
                exact(mid(in,tocol(rook(pos,in),3),1),{"r","q"}),
                exact(mid(in,tocol(knight(pos,in),3),1),"n"),
                exact(mid(in,tocol(king(pos,in),3),1),"k"),
                exact(mid(parse(pos,in),find("X",parse(pos,in))-{23,21},1),"p")
            )))),
        
        generatemoves,lambda(in,pos,parse,
            let(curpiece,mid(in,pos,1),
                moves,tocol(
                    if(or(curpiece=" ",not(iswhite(curpiece))),NA(),
                    if(curpiece="b",bishop(pos,in),
                    if(curpiece="r",rook(pos,in),
                    if(curpiece="n",knight(pos,in),
                    if(curpiece="q",queen(pos,in),
                    if(curpiece="p",pawn(pos,in),
                    if(curpiece="k",king(pos,in),NA()))))))),3),
                if(rows(moves),
                    if(parse,
                        map(moves,lambda(move,
                            replace(
                            if(and(curpiece="p",move=ep),
                                replace(replace(in,move,1,curpiece),move+8,1," "),
                                replace(in,move,1,curpiece)
                            ),pos,1," ")&"/"&curpiece&"/"&move)),
                        moves
                    ),
                    NA()
                )
            )
        ),

        evaluate,lambda(in,
            let(string, regexreplace(join(,regexreplace(flip(left(in,64)),"[^"&{"P","N","B","R","Q","K","p","n","b","r","q","k"}&"]","0")),"[^0]","1")&--whitesturn&"0000"&rept("0",7)&rept("0",4),
                layers, reduce(--mid(string,sequence(784),1),sequence(7)-1,lambda(a,b,mmult(offset(nnweightsv1!A1,b*784,,784,784),(a+abs(a))/2))),
                sumproduct((layers+abs(layers))/2,tocol(nnweightsv1!A5489:ADD5489))
            )
        ),
        
        issafe,lambda(move,
            let(boardstate,single(split(move,"/")),
                not(isattacked(find("K",boardstate),boardstate))
            )
        ),

        safemoves,lambda(moves,
            ifna(filter(moves,map(moves,lambda(move,issafe(move)))),board)
        ),
        
        rankmoves,lambda(moves,
            if(single(moves)=board,{board,"CHECKMATE"},sort(map(moves,lambda(move,{move,evaluate(move)})),2,))
        ),

        movesfromfen,lambda(
            tocol(map(sequence(64),lambda(square,torow(generatemoves(board,square,1)))),3)
        ),

        out, rankmoves(safemoves(movesfromfen())),

        out
    )
)
