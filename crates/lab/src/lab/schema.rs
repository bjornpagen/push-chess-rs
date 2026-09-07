//! A finite relational theory of campaigns, boards, transitions and searches.
//! No serialized configuration, FEN, JSON, PV vectors or diagnostic blobs.
//! Missing optional observations are absent facts, not sentinel values.
bumbledb::schema! {
    pub TrainingGround;
    closed relation Split as SplitId = { Train, Validation, Test, Evaluation };
    closed relation Purpose as PurposeId = { Corpus, Arena };
    closed relation BudgetKind as BudgetKindId = { Nodes, Time };
    closed relation RunStatus as RunStatusId = { Finished, Interrupted, Failed };
    closed relation Ending as EndingId = { Mate, Stalemate, FiftyMove, Repetition, PlyLimit, Interrupted };
    closed relation Side as SideId = { White, Black };
    closed relation PieceKind as PieceKindId { promotes:bool } = {
        Pawn {promotes:false}, Knight {promotes:true}, Bishop {promotes:true},
        Rook {promotes:true}, Queen {promotes:true}, King {promotes:false},
    };
    closed relation Wing as WingId = { KingSide, QueenSide };
    closed relation Route as RouteId = { Direct, RankFirst, FileFirst };
    closed relation MoveKind as MoveKindId = { Ordinary, Castle, EnPassant, Promotion };
    closed relation WhiteOutcome as WhiteOutcomeId { value:i64 } = {
        Win {value:1}, Draw {value:0}, Loss {value:-1},
    };
    relation Run {
        id:u64 as RunId, binary:bytes<32>, rules:str, started_us:u64,
        pairs:u64, workers:u64, max_plies:u64, opening_plies:u64, seed:u64,
        purpose:u64 as PurposeId, budget:u64 as BudgetKindId,
        max_seconds:u64, max_bytes:u64,
    }
    relation NodeBudget {run:u64 as RunId, nodes:u64}
    relation TimeBudget {run:u64 as RunId, microseconds:u64}
    relation RunEnd {run:u64 as RunId, status:u64 as RunStatusId, finished_us:u64}
    relation RunFailure {run:u64 as RunId, message:str}
    relation Engine {
        id:bytes<32> as EngineId, binary:bytes<32>, name:str, lineage:str, hypothesis:str,
        neural_accumulator:bool, neural_evaluation:bool,
    }
    relation EngineNetwork {engine:bytes<32> as EngineId, fingerprint:u64}
    relation Entrant {run:u64 as RunId, slot:u64, engine:bytes<32> as EngineId}
    relation Pair {run:u64 as RunId, index:u64, first:u64, second:u64, opening:bytes<32>}
    relation Setup {
        id:bytes<32> as SetupId, side:u64 as SideId, halfmove_clock:u64, fullmove_number:u64,
    }
    relation SetupPiece {setup:bytes<32> as SetupId, square:u64, side:u64 as SideId, kind:u64 as PieceKindId}
    relation SetupCastle {setup:bytes<32> as SetupId, side:u64 as SideId, wing:u64 as WingId}
    relation SetupEp {setup:bytes<32> as SetupId, square:u64}
    relation Game {
        run:u64 as RunId, index:u64, pair:u64, swapped:bool, setup:bytes<32> as SetupId,
        trajectory:bytes<32>, split:u64 as SplitId, ending:u64 as EndingId, plies:u64,
    }
    relation GameResult {run:u64 as RunId, game:u64, outcome:u64 as WhiteOutcomeId}
    relation Position {
        run:u64 as RunId, game:u64, ply:u64, side:u64 as SideId,
        halfmove_clock:u64, fullmove_number:u64, hash:u64,
    }
    relation CastlingRight {run:u64 as RunId, game:u64, ply:u64, side:u64 as SideId, wing:u64 as WingId}
    relation EnPassant {run:u64 as RunId, game:u64, ply:u64, square:u64}
    relation Action {
        id:u64 as ActionId, from:u64, to:u64, route:u64 as RouteId, stop:u64, kind:u64 as MoveKindId,
    }
    relation ActionPromotion {action:u64 as ActionId, piece:u64 as PieceKindId}
    relation Move {run:u64 as RunId, game:u64, ply:u64, action:u64 as ActionId}
    relation PieceRemoved {run:u64 as RunId, game:u64, ply:u64, square:u64, side:u64 as SideId, kind:u64 as PieceKindId}
    relation PiecePlaced {run:u64 as RunId, game:u64, ply:u64, square:u64, side:u64 as SideId, kind:u64 as PieceKindId}
    relation Analysis {
        run:u64 as RunId, game:u64, ply:u64, score_stm:i64, complete:bool,
        nodes:u64, depth:u64, seldepth:u64, wall_us:u64, reported_us:u64,
        qnodes:u64, tt_hits:u64, pv_length:u64,
    }
    relation PvStep {run:u64 as RunId, game:u64, ply:u64, step:u64, action:u64 as ActionId}
    relation ProofSearch {run:u64 as RunId, game:u64, ply:u64, nodes:u64}
    relation MateProof {run:u64 as RunId, game:u64, ply:u64, plies:u64}

    Run(id)->Run;
    Run(purpose)<=Purpose(id);
    Run(budget)<=BudgetKind(id);
    NodeBudget(run)->NodeBudget;
    TimeBudget(run)->TimeBudget;
    NodeBudget(run)==Run(id | budget==Nodes);
    TimeBudget(run)==Run(id | budget==Time);
    RunEnd(run)->RunEnd;
    RunEnd(run)<=Run(id);
    RunEnd(status)<=RunStatus(id);
    RunFailure(run)->RunFailure;
    RunFailure(run)==RunEnd(run | status==Failed);
    Entrant(run,slot)->Entrant;
    Entrant(run,engine)->Entrant;
    Entrant(run)<=Run(id);
    Entrant(engine)<=Engine(id);
    Engine(id)->Engine;
    Engine(binary,name)->Engine;
    EngineNetwork(engine)->EngineNetwork;
    EngineNetwork(engine)==Engine(id | neural_accumulator==true);
    Pair(run,index)->Pair;
    Pair(run,first)<=Entrant(run,slot);
    Pair(run,second)<=Entrant(run,slot);
    Setup(id)->Setup;
    Setup(side)<=Side(id);
    SetupPiece(setup,square)->SetupPiece;
    SetupPiece(setup)<=Setup(id);
    SetupPiece(side)<=Side(id);
    SetupPiece(kind)<=PieceKind(id);
    SetupCastle(setup,side,wing)->SetupCastle;
    SetupCastle(setup)<=Setup(id);
    SetupCastle(side)<=Side(id);
    SetupCastle(wing)<=Wing(id);
    SetupEp(setup)->SetupEp;
    SetupEp(setup)<=Setup(id);
    Game(run,index)->Game;
    Game(run,pair,swapped)->Game;
    Game(run,pair)<=Pair(run,index);
    Game(setup)<=Setup(id);
    Game(split)<=Split(id);
    Game(ending)<=Ending(id);
    GameResult(run,game)->GameResult;
    GameResult(run,game)==Game(run,index | ending=={Mate,Stalemate,FiftyMove,Repetition});
    GameResult(outcome)<=WhiteOutcome(id);
    Position(run,game,ply)->Position;
    Position(run,game)<=Game(run,index);
    Game(run,index,plies)<=Position(run,game,ply);
    Position(side)<=Side(id);
    CastlingRight(run,game,ply,side,wing)->CastlingRight;
    CastlingRight(run,game,ply)<=Position(run,game,ply);
    CastlingRight(side)<=Side(id);
    CastlingRight(wing)<=Wing(id);
    EnPassant(run,game,ply)->EnPassant;
    EnPassant(run,game,ply)<=Position(run,game,ply);
    Action(id)->Action;
    Action(route)<=Route(id);
    Action(kind)<=MoveKind(id);
    ActionPromotion(action)->ActionPromotion;
    ActionPromotion(action)==Action(id | kind==Promotion);
    ActionPromotion(piece)<=PieceKind(id | promotes==true);
    Move(run,game,ply)->Move;
    Move(run,game,ply)<=Position(run,game,ply);
    Move(action)<=Action(id);
    PieceRemoved(run,game,ply,square)->PieceRemoved;
    PieceRemoved(run,game,ply)<=Move(run,game,ply);
    PieceRemoved(side)<=Side(id);
    PieceRemoved(kind)<=PieceKind(id);
    PiecePlaced(run,game,ply,square)->PiecePlaced;
    PiecePlaced(run,game,ply)<=Move(run,game,ply);
    PiecePlaced(side)<=Side(id);
    PiecePlaced(kind)<=PieceKind(id);
    Analysis(run,game,ply)->Analysis;
    Analysis(run,game,ply)<=Move(run,game,ply);
    PvStep(run,game,ply,step)->PvStep;
    PvStep(run,game,ply)<=Analysis(run,game,ply);
    PvStep(action)<=Action(id);
    ProofSearch(run,game,ply)->ProofSearch;
    ProofSearch(run,game,ply)<=Analysis(run,game,ply);
    MateProof(run,game,ply)->MateProof;
    MateProof(run,game,ply)<=ProofSearch(run,game,ply);
    // Contiguous exact cardinalities are checked at the atomic replay boundary;
    // the theory supports dependent upper bounds but not dependent floors.
    Game(run,index)<={0..plies}Move(run,game);
    Analysis(run,game,ply)<={0..pv_length}PvStep(run,game,ply);
}
