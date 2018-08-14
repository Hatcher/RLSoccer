/*
  State Representation
 * We want to represent the
 state and pass that to the agent,
   We can use the distances:
   We will refer to our players as P and the oppoing players as O
   - P_1 will be the player with the ball
   1. Distance P_1 to each P_i
   - We can use some kind of center point as the area from which we triangulate and try to setup
   - This way the players are passing around and moving within a defined space, but if we use a fixed position would that effect the rewards calculations?
   2. Distance from each player to some kind of center position
   - We want to provide information about the distance for opponents
   3. Distance for each P_i to each O_i
   - We also provide the minimum angles between players with a vertex of P_1
   4. Angle for P_i(P_1, O_i)
 */

 /*
 Actions
 * We want our actoins to be highlevel choices that the agent makes? Or should it simply be 1-10 for speed and -10 to 10 for phi?
 * We should define a few actions that are high level and make use of the state
  If its highlevel choices we can preprogram:
  1. Hold Ball
  2. Get Open
  3. Pass Ball(P)
  4. Go To Ball
  5. Block Pass(P) This moves between the player with the ball and another player that we expect to receive it.
 */

 /*
 Rewards
 * The reward for a given time step can be the rewards for the previous time step minus the reward at this time step?
  We can reward based on:
  1. Ball goes in goal
  2. Ball distance to goal is smaller
  3. Other team has ball is negative reward
  4. Change of possession, although if the ball is in the middle of the field after a goal it might negatively effect the reward for Scoring.
  5. Ball moving to the our half of the field would be bad? but accounted for in distance....
 */


import java.util.*;
import java.awt.geom.*;
import java.lang.Math;


public class teamRL implements TeamController {

    public enum Actions {
        HOLD_BALL,      // offensive:0
        //GET_OPEN,       // offensive:1 Is considered part of hold ball now
        PASS_BALL,      // offensive:1
        GO_TO_BALL,     // defensive:2
        BLOCK_PASS,     // defensive:3
    };

    SensorPack sensors;               // The sensor pack handed to us.
    int numPlayers;                   // How many players/team.
    ArrayList<Player> players;        // List of our players.
    int myteam = -1;                  // Our team number (0 or 1).
    int oppteam = -1;                 // Opposing team number.
    boolean onLeft;                   // Are we on the left or right?

    boolean debug = false;            // If we want to turn on debugging.

    /* for REWARD FUNCTION *********************************************/
    private final double DISCOUNT_FACTOR = 0.9;
    private int discountPow;

    double epsilon = 0.10;

    Point2D.Double myGoalPoint;
    Point2D.Double oppGoalPoint;

    Point2D.Double p1BlockingPosition;
    Point2D.Double p2BlockingPosition;
    // RL stuff
    Soccer parent;
    RLState state;
    int actionTaken;
    Map<RLState, Map<Integer, Double>> savMap;    //State Action Value map
    Map<RLState, Map<Integer, Integer>> sacMap;   //State Action Count map
    ArrayList<RLStateAction> saList;
    double centerX, centerY;

    //My score + opponent scores
    private int myPrevGoals;
    private int oppPrevGoals;
    private int myGoalsScored = 0;
    private int oppGoalsScored = 0;

    //Action flags
    int receiving = 0;
    double prevBallX = 50.0;
    double prevBallY = 20.0;

    public RLState initState() {
        RLState newState = new RLState();
        newState.ballState = 0;
        newState.teamHasBall = false;
        newState.teamStatus = new int[]{-1, -1, 0};
        int i = 0;
        for (Player p : players) {
          int gridNum;
          if (p.y < 13.33) {
            gridNum = 0;
          } else if (p.y < 26.67) {
            gridNum = 3;
          } else {
            gridNum = 6;
          }

          if (p.x >= 33.33) {
            gridNum += 1;
          } else if (p.x >= 66.66) {
            gridNum += 2;
          }
          newState.teamLocations[i++] = gridNum;
        }

        // get opp locations
        i = 0;
        for (Point2D.Double p : opposingPlayerPositions()) {
          int gridNum;
          if (p.y < 13.33) {
            gridNum = 0;
          } else if (p.y < 26.67) {
            gridNum = 3;
          } else {
            gridNum = 6;
          }

          if (p.x >= 33.33) {
            gridNum += 1;
          } else if (p.x >= 66.66) {
            gridNum += 2;
          }
          newState.oppLocations[i++] = gridNum;
        }
        return newState;
    }

    public ArrayList<RLState> initStateSpace() {
      ArrayList<RLState> stateSpace = new ArrayList<RLState>();
      int count = 0;
      for (int ballScored = -1; ballScored <= 1; ballScored++) {
        for (int hasBall = 0; hasBall <= 1; hasBall++) {
          int[] teamMemberStatuses = new int[3];
          for (int tm1s = -1; tm1s <= 1; tm1s++) {
              for (int tm2s = -1; tm2s <= 1; tm2s++) {
                  for (int tm3s = -1; tm3s <= 1; tm3s++) {
                      teamMemberStatuses = new int[]{tm1s, tm2s, tm3s};
                      int[] teamMemberLocations = new int[2];
                      for (int tm1l = 0; tm1l < 9; tm1l++) {
                          for (int tm2l = 0; tm2l < 9; tm2l++) {
                              teamMemberLocations = new int[]{tm1l, tm2l};
                              int[] oppMemberLocations = new int[2];
                              for (int om1l = 0; om1l < 9; om1l++) {
                                  for (int om2l = 0; om2l < 9; om2l++) {
                                      oppMemberLocations = new int[]{om1l, om2l};
                                      RLState newState = new RLState();
                                      newState.ballState = ballScored;
                                      newState.teamHasBall = (hasBall == 1);
                                      newState.teamStatus = teamMemberStatuses;
                                      newState.teamLocations = teamMemberLocations;
                                      newState.oppLocations = oppMemberLocations;
                                      count++;
                                  }
                              }
                          }
                      }
                  }
              }
          }
        }
      }
      return stateSpace;
    }

    // This function should take in a state based on the current frame in
    // soccer.java and the savMap to determine an action taken
    public int chooseMove(RLState state, double epsilon,
        HashMap<RLState, HashMap<Integer, Double>> savMap,
        HashMap<RLState, HashMap<Integer, Integer>> sacMap) {

        // get map of action-value pairs given the savMap and state
        Map<Integer, Double> avMap = savMap.get(state);
        if (state.teamHasBall) {
            double hbav = avMap.get(Actions.HOLD_BALL.ordinal());
            double pbav = avMap.get(Actions.PASS_BALL.ordinal());

            // Check if HOLD_BALL av is better than PASS_BALL
            return (hbav > pbav ? Actions.HOLD_BALL.ordinal() : Actions.PASS_BALL.ordinal());
        } else {
            double gtbav = avMap.get(Actions.GO_TO_BALL.ordinal());
            double bpav = avMap.get(Actions.BLOCK_PASS.ordinal());
            return (gtbav > bpav ? Actions.GO_TO_BALL.ordinal() : Actions.BLOCK_PASS.ordinal());
        }
    }

    public double getReward(RLState newState) {
        return newState.ballState;
    }

    public void setState(RLState state) {
      this.state = state;
    }

    public void init(SensorPack sensors,int numPlayers, int myteam, boolean onLeft) {
        this.sensors = sensors;
        this.numPlayers = numPlayers;
        this.myteam = myteam;
        this.onLeft = onLeft;
        players = new ArrayList<Player> ();
        saList = new ArrayList<RLStateAction>();
        if(onLeft){
          p1BlockingPosition = new Point2D.Double(25.0, 35.0);
          p2BlockingPosition = new Point2D.Double(25.0, 5.0);
          oppGoalPoint = new Point2D.Double(100.0, 20.0);
          myGoalPoint = new Point2D.Double(0.0, 20.0);
        }else{
          p1BlockingPosition = new Point2D.Double(75.0, 35.0);
          p2BlockingPosition = new Point2D.Double(75.0, 5.0);
          oppGoalPoint = new Point2D.Double(0.0, 20.0);
          myGoalPoint = new Point2D.Double(100.0, 20.0);
        }
    }

    // End RL stuff

    public String getName () {
      return "Pepe is best team";
    }


    public void init (Soccer parent, SensorPack sensors, int numPlayers, int myteam, boolean onLeft) {
      this.parent = parent;
      this.sensors = sensors;
      this.numPlayers = numPlayers;
      this.myteam = myteam;
      this.onLeft = onLeft;
      this.centerX = 50.;
      this.centerY = 20.;
      players = new ArrayList<Player> ();
      if (myteam == 0) {                //Determines which team we are on
        this.oppteam = 1;
      } else {
        this.oppteam = 0;
      }


      if(onLeft){
        p1BlockingPosition = new Point2D.Double(25.0, 35.0);
        p2BlockingPosition = new Point2D.Double(25.0, 5.0);
        oppGoalPoint = new Point2D.Double(100.0, 20.0);
        myGoalPoint = new Point2D.Double(0.0, 20.0);
      }else{
        p1BlockingPosition = new Point2D.Double(75.0, 35.0);
        p2BlockingPosition = new Point2D.Double(75.0, 5.0);
        oppGoalPoint = new Point2D.Double(0.0, 20.0);
        myGoalPoint = new Point2D.Double(100.0, 20.0);
      }
      state = initState();
    }

    public void init (int playerNum, double initX, double initY, double initTheta)
    {
      Player p = new Player ();
      p.playerNum = playerNum;
      p.x = initX;
      p.y = initY;
      p.theta = initTheta;
      players.add (p);
    }

    public void setAction(int action) {
      this.actionTaken = action;
    }

    public void move ()
    {
      updateLocations ();
      int t = (int) sensors.getCurrentTime ();
      int result = 0;
      if(UniformRandom.uniform() < epsilon){
        result = pickRandomMove();
      }else{
        //Pick the move with the highest value
        switch (Actions.values()[actionTaken]) {
          case HOLD_BALL:
            holdBall();
            break;
          case PASS_BALL:
            if(sensors.isBallHeld() && sensors.ballHeldByTeam() == myteam){
              if(sensors.ballHeldByPlayer() == 0){
                passBall(getPlayer(sensors.ballHeldByPlayer()), getPlayer(1));
              }else{
                passBall(getPlayer(sensors.ballHeldByPlayer()), getPlayer(0));
              }
            }
            break;
          case GO_TO_BALL:
            moveToBall();
            break;
          case BLOCK_PASS:
            blockPass();
            break;
        }
      }
      // Needs to be at the bottom of the move method.
      prevBallY = sensors.getBallY();
      prevBallX = sensors.getBallX();
    }

    public int pickRandomMove(){
      int move = UniformRandom.uniform(0, 3);
      if(move == 0){
        holdBall();
      }else if(move == 1){
        if(sensors.isBallHeld() && sensors.ballHeldByTeam() == myteam){
          if(sensors.ballHeldByPlayer() == 0){
            passBall(getPlayer(sensors.ballHeldByPlayer()), getPlayer(1));
          }else{
            passBall(getPlayer(sensors.ballHeldByPlayer()), getPlayer(0));
          }
        }
      }else if(move == 2){
        moveToBall();
      }else if(move == 3){
        blockPass();
      }
      return move;
    }


    public double getControl (int p, int i)
    {
      Player player = getPlayer (p);
      if (i == 1) {
        return player.vel;
      }
      else {
        return player.phi;
      }
    }


  public boolean triesKick (int p){
    // If we have the ball, we're going to kick.
    Player player = players.get(p);
    if(player.hasBall && player.tryKick){
      return true;
    }
    return false;
  }

    //TODO
    public void kickSuccessful (int p)
    {
    }


    public boolean triesGrab (int p)
    {
      return false;
    }


    public void grabSuccessful (int p)
    {
    }


    void updateLocations ()
    {
      for (Player p: players) {
        p.tryKick = false;
        p.x = sensors.getX (myteam, p.playerNum);
        p.y = sensors.getY (myteam, p.playerNum);
        p.theta = sensors.getTheta (myteam, p.playerNum);
        p.distToBall = distance (sensors.getBallX(), sensors.getBallY(), p.x,p.y);
        if ( (sensors.isBallHeld()) && (sensors.ballHeldByTeam() == myteam) && p.playerNum == sensors.ballHeldByPlayer()) {
          p.hasBall = true;
        }
        else {
          p.hasBall = false;
        }
      }
    }

    Player getPlayer (int k)
    {
      for (Player player: players) {
        if (player.playerNum == k) {
          return player;
        }
      }
      return null;
    }


    double distance (double x1, double y1, double x2, double y2)
    {
      return Math.sqrt ( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
    }

    public void startDebug ()
    {
      debug = true;
    }

/***************** Helper functions
* 1. Facing, returns boolean if generally facing the desired direction
* n.

/*
* Checks if facing generally towards the point
*/
public Boolean facing(Player p, Point2D.Double point){
  boolean result = false;
  double desiredTheta = sensors.getAngle(p.x, p.y, point.x, point.y);
  if(debug){
    System.out.println( "My theta is "+ p.theta + " my desiredTheta is " + desiredTheta );
  }
  if(p.theta <= desiredTheta+.24 && p.theta >= desiredTheta-.24){
    result = true;
  }
  if((desiredTheta >= 6.05 && p.theta <= 0.15) || (desiredTheta <=0.15 && p.theta >= 6.05)){
    if(debug){
      System.out.println( " I am in facing and i am facing my target but its all the way around" + p.playerNum );
    }
    result = true;
  }
  if(debug){
    System.out.println( " I am in facing and i am not looking at my target " + p.playerNum );
  }
  return result;
}
/*
* Turn the player to face the desired position.
*/
public void facePosition(Player p, double desiredX, double desiredY){
    double playerTheta = p.theta;
    double desired = sensors.getAngle( desiredX, desiredY, p.x, p.y);
    double distanceToPos = distance(p.x, p.y, desiredX, desiredY);
    if(debug){
      //System.out.println( "Desired theta is " + desired + " actual Theta is " +p.theta + " player num is " + p.playerNum );
    }
    if(Math.abs(desired - p.theta) > Math.PI){
      if(desired > p.theta){
        if(debug){
          System.out.println( "Turning right to get to" + desired+ " player num is " + p.playerNum);
        }
        p.phi = 10;
      }else{
        p.phi = -10;
      }
    }else{
      if(desired > p.theta){
        p.phi = -10;
      }else{
        if(debug){
          System.out.println( "Turning right player num is " + p.playerNum);
        }
        p.phi = 10;
      }
    }
    if(distanceToPos > 20){
      p.vel = 10;
    }if(distanceToPos > 10){
      p.vel = 5;
    }else {
      p.vel = 2;
    }
  }
/*
* We want to move to the position(x,y)
*/
public Boolean moveTo(Player player, double x, double y){
  int pNum = player.playerNum;
  if(debug){
    System.out.println( "I "+ player.playerNum+ " am moving to " + x +", "+y );
  }
  if(distance(x, y, player.x, player.y) < 2){
    player.vel = 0;
    return true;
  }else{
    Point2D.Double point = new Point2D.Double(x, y);
    if(facing(player, point)){ //We are facing the correct way
      player.vel = 10;  //Move at max speed
      player.phi = 0;
    }else{
      facePosition(player, x, y); //Turn to face the position
    }
  }
  return false;
}

/*
* Returns a boolean of the player being near their blocking position
*/
public boolean isInBlocking(Player p){
  if(p.playerNum == 0){
    if(distance(p.x, p.y, p1BlockingPosition.x, p1BlockingPosition.y) < 6){
      return true;
    }
  }else if(p.playerNum == 1){
    if(distance(p.x, p.y, p2BlockingPosition.x, p2BlockingPosition.y) < 6){
      return true;
    }
  }
  return false;
}

/*
* Pass to the player
*/
public void passToTeammate(Player p, Player receiving){
  //We want to pass infront of the teammate
  int passingBall, receivingBall;
  double futureX;
  if(onLeft){
    futureX = (receiving.x + 100)/2;
  }else{
    futureX = receiving.x/2;
  }
  double futureY = receiving.y + .1  * receiving.vel * Math.sin(receiving.theta);
  if(distance(futureX, futureY, receiving.x, receiving.y) > 15){
    futureX = receiving.x + .1  * receiving.vel * Math.cos(receiving.theta);
  }
  Point2D.Double receiver = new Point2D.Double(futureX, futureY);
  if(facing(p, receiver) && distance(p.x, p.y, receiving.x, receiving.y) > 15){
    passingBall = p.playerNum; //Set our passingball number to the p's number
    receivingBall = receiving.playerNum;
    if(debug){
      System.out.println( "I, player: " + p.playerNum+ " am going to pass the ball to: " + receiving.playerNum + " at a distance of " + distance(p.x, p.y, receiving.x, receiving.y) );
    }
    p.tryKick = true;
    triesKick(p.playerNum);
  }else if(facing(p, receiver)){
    p.phi = 0;
  }else if(distance(p.x, p.y, receiving.x, receiving.y) < 15){
    if(debug){
      System.out.println( "I, player: " + p.playerNum+ " cant pass to " + receiving.playerNum + " at a distance of " + distance(p.x, p.y, receiving.x, receiving.y) );
    }
    //Too close to successfully pass
    moveTo(p, p.x, p.y-4);
  }
}
/*rets bool=true if we have ball. else false */
 //accounts for passing because we update
public boolean weHavePossession(){
  boolean isBallHeld = sensors.isBallHeld();
  if(isBallHeld) {
    if(sensors.ballHeldByTeam() == myteam) {
      return true;
    }
    return false;
  }
  return false;
}

public void setMyGoalsScored(int newScore) {
  myPrevGoals = myGoalsScored;
  myGoalsScored = newScore;
}

public void setOppGoalsScored(int newScore) {
  oppPrevGoals = oppGoalsScored;
  myGoalsScored = newScore;
}

/* returns true if opp scored in prev state
*/
public boolean theyScoredGoal(){
  return oppPrevGoals != oppGoalsScored;
}
/* returns true if opp scored in prev state
*/
public boolean weScoredGoal(){
  return myPrevGoals != myGoalsScored;
}

/*
* Return ths closest player number of the opposing player to the ball
*/
public int getOpposingPlayerClosestToBall(){
  double shortestDist = Double.MAX_VALUE;
  int playerNumber = 0;
  for(int i = 0; i < numPlayers; i++){
    double testDistance = distance(sensors.getBallX(), sensors.getBallY(), sensors.getX(oppteam, i), sensors.getY(oppteam, i));
    if(testDistance < shortestDist) {
      shortestDist = testDistance;
      playerNumber = i;
    }
  }
  return playerNumber;
}
/*
* Returns an arraylist of point2d for opposing player positions
*/
public ArrayList<Point2D.Double> opposingPlayerPositions(){
  ArrayList<Point2D.Double> list = new ArrayList<Point2D.Double>(numPlayers);
  for(int i = 0 ; i < numPlayers; i++){
    double x = sensors.getX(oppteam, i);
    double y = sensors.getY(oppteam, i);
    Point2D.Double point = new Point2D.Double(x, y);
    list.add(point);
  }
  return list;
}

/*
* Returns boolean if an opposing player is close to them
*/
public boolean opposingPlayerCloseToMe(Point2D.Double point, int range){
  for(int i = 0; i < numPlayers; i++){
    double dist = distance(point.x, point.y, sensors.getX(oppteam, i), sensors.getY(oppteam, i));
    if( dist < range){
      return true;
    }
  }
  return false;
}

/*
* Returns a 2D point of the provided player
*/
public Point2D.Double getPoint(Player p){
  Point2D.Double point = new Point2D.Double(p.x, p.y);
  return point;
}

/*
* Calculates the distance to the line between two points
*/
public double pointDistanceToLine(Point2D.Double point1, Point2D.Double point2, Point2D.Double target){
  double theLine = distance(point1.x, point1.y, point2.x, point2.y);
  double oppDistToLine =   Math.abs((point2.x - point1.x)*( point1.y - target.y) - (point1.x - target.x)*(point2.y - point1.y))/theLine;
  return oppDistToLine;
}

/*
* Calculates the angle between three points
*/
public double angleBetweenPoints(Point2D.Double vertex, Point2D.Double point1, Point2D.Double point2){
  double numerator = Math.pow(distance(vertex.x,vertex.y, point1.x, point1.y),2) + Math.pow(distance(vertex.x,vertex.y, point2.x, point2.y), 2) - Math.pow(distance( point1.x, point1.y,  point2.x, point2.y), 2);
  double denominator = 2*distance(vertex.x,vertex.y, point1.x, point1.y)*distance(vertex.x,vertex.y, point2.x, point2.y);
  return Math.acos(numerator/denominator);
}
/*
* Returns a boolean if we are between the ball and our goal
*/
public boolean betweenBallAndGoal(Player p){
  if(distance(myGoalPoint.x, myGoalPoint.y, p.x, p.y) < distance(myGoalPoint.x, myGoalPoint.y, sensors.getBallX(), sensors.getBallY())){
    return true;
  }
  return false;
}

public Point2D.Double getBallPoint() {
  return new Point2D.Double(prevBallX, prevBallY);
}
// Calculate a point to get open at
//An episode will be when a goal is scored, when a player has a shot the other player moves to the middle of the map
// This will be hard coded by yours truly.

/*
* Returns boolean if the player can kick to this position
*/
public boolean hasLineOfSight(Player p, Point2D.Double target){
    //If the player is within 5 of the line, say no
    for(int i = 0; i < players.size(); i++){
      //Should check if we have any other players within any players within a certain range of our line to goal.
      //Use 2 points, our current position to the goal position....
      double oppX = sensors.getX(oppteam, i);
      double oppY = sensors.getY(oppteam, i);
      Point2D.Double opponent = new Point2D.Double(oppX, oppY);
      Point2D.Double playerPoint = new Point2D.Double(p.x, p.y);
      double oppDistToLine =  pointDistanceToLine(playerPoint, target, opponent);
      /*if(debug){
        System.out.println( " The oppositions distance to our line is " + oppDistToLine );
      }*/
      if(oppDistToLine < 3 && oppDistToLine > 0.0){
        if(onLeft){
          if(oppX > p.x){
            return false;
          }
        }else{
          if(oppX < p.x){
            return false;
          }
        }
      }
    }
    return true;
  }

/*
* The player needs to move away from the opponents and towards the goal if he is away from the opponents
*/
public void getOpen(Player p){
  double desiredX = 0.0;
  double desiredY = 5.0;  //We want to move to a position that is either low or high
  if(!opposingPlayerCloseToMe(getPoint(p), 8)){
    //Need to move towards the goal.
    desiredX = (oppGoalPoint.x + p.x)/2; //We want to move half way between the goal and our current position
    if(sensors.getBallY() > 33.0){
      desiredY = 20.0;
    }else if(sensors.getBallY() > 15.0){
      desiredY = 5.0;
    }else{
      desiredY = 35.0;
    }
  }else{
    //Need to calculate a position that is a certain distance away from the player but not towards our goal
    if(distance(sensors.getBallX(), sensors.getBallY(), oppGoalPoint.x, oppGoalPoint.y) > 25){    //If the ball is 20 away from the opponents goal
      //Move to an x that is 30 away from the opponents goal
      desiredX = (oppGoalPoint.x + 40.0)/2;
      //Move to a y that is in a line of sight to the player with the ball
      //find line of sight of ball - generate line of sight
    }else{    //If the ball is more than 20 away from the goal
      //Move to an x that is 15 away from the opponents goal
      desiredX = (oppGoalPoint.x + 30)/2;
      //Move to a y that is opposite of the ball
    }
    if(sensors.getBallY() > 33.0){ //We want to go to the mid
      desiredY = UniformRandom.uniform(15.0, 25.0);
    }else if(sensors.getBallY() > 15.0){  //we want to go low
      desiredY = UniformRandom.uniform(0.0, 15.0);
    }else{  //We want to go high
      desiredY = UniformRandom.uniform(26.0,40.0);
    }
  }
  moveTo(p, desiredX, desiredY);
}
/*
* If the player has a shot on the goal return a boolean
*/
public boolean hasShot(Player p){
    //Check if we have any other players in between us and the goal
    double gx = oppGoalPoint.x;
    double gy = oppGoalPoint.y;
    Point2D.Double bottomOfGoal = new Point2D.Double(gx, gy-7);
    Point2D.Double topOfGoal = new Point2D.Double(gx, gy+7);
    Point2D.Double playerPos = new Point2D.Double(p.x, p.y);
    double distToGoal = pointDistanceToLine(bottomOfGoal, topOfGoal, playerPos);
    if(distToGoal > 40){
      //System.out.println( " no shot " + distToGoal );
      return false;
    }
    return hasLineOfSight(p, oppGoalPoint);
  }

/*
* Checks if a player is close by
*/
  public boolean isOpen(Player p, int range){
    for(int i = 0; i < numPlayers; i++){
      double dist = distance(p.x, p.y, sensors.getX(oppteam, i), sensors.getY(oppteam, i));
      if( dist < range){
        return true;
      }
    }
    return false;
  }

/***************** Available actions
* 1. Move to ball, allows the agent to move towards the ball, stationary or moving
* 2. Hold Ball, stand still while holding the ball, if on defense, we...
* 3. Pass Ball(p), pass the ball to player p
* 4. Get Open, move to a position that is away from opponents REMOVED
* 5. Block Pass(p)
*/
/*
* 1. Moves the player towards the ball
* - Within the range grab the ball
*/
public void moveToBall(){
  for(Player p : players){
    double ballX = sensors.getBallX();
    double ballY = sensors.getBallY();
    double ballV = distance(ballX, ballY, this.prevBallX, this.prevBallY)/.1;
    double futureBallX = ballX + .1 * ballV * Math.cos(sensors.getBallTheta());
    double futureBallY = ballY + .1 * ballV * Math.sin(sensors.getBallTheta());
    moveTo(p, futureBallX, futureBallY);
  }
}

/*
* 2. Hold ball, stand still while holding the ball
*/
public void holdBall(){
  //Absorbs the get open function
  //if not p with ball:
  int playerWithBallNum = 0;
  if(sensors.isBallHeld()){
    playerWithBallNum = sensors.ballHeldByPlayer();

  for(int i = 0; i < numPlayers; i++){
    //if player w/ball
    if(i == playerWithBallNum){
      Player p = getPlayer(playerWithBallNum);
      //When holding the ball move towards the goal at slow pace
      moveTo(p, oppGoalPoint.x, oppGoalPoint.y);
      if(p.vel == 10){  //If the player is facing the correct direction, move slowly
        p.vel = 3;
      }
    }else{          //not have ball
      Player p = getPlayer(i);
      getOpen(p);   //get open why is this underlined?
    }
  }
}
}
/*
* 3. Pass the ball to player r
*/
public void passBall(Player p, Player r){
  passToTeammate(p, r);
}

/*
* 4.block pass
*_____assuming other TEAM has ball____________
*calculate the midpoint between:
*      player with ball, and goal
* then player with ball, and other player
* for each midpoint, calculate our players distance to that midpoint.
* For each combination of distances to midpoints, calculate the sum.
* for the lowest sum, send the players to those midpoints.
*/
public void blockPass(){
  double theX = 50.0;
  double theY = 20.0;
  int oppPlayer = 0;

  //If the ball is held, use that players location,otherwise use the balls location
  if(sensors.ballHeldByPlayer() == -1){ //If the ball is being passed.
    theX = sensors.getBallX();
    theY = sensors.getBallY();
    oppPlayer = getOpposingPlayerClosestToBall();
  }else{
    oppPlayer = sensors.ballHeldByPlayer();
    theX = sensors.getX(oppteam, oppPlayer);
    theY = sensors.getY(oppteam, oppPlayer);
  }

  //The midpoint between the goal and the position
  double ballAndGX = (myGoalPoint.x + theX)/2;
  double ballAndGY = (myGoalPoint.y + theY)/2;
  Point2D.Double goalMidPoint = new Point2D.Double(ballAndGX, ballAndGY);

  //The midpoint between the players and the position
  ArrayList<Point2D.Double> playerLoc = opposingPlayerPositions();
  ArrayList<Point2D.Double> midPoints = new ArrayList<Point2D.Double>(numPlayers);
  for(int i = 0; i < playerLoc.size(); i++){
    if(i != oppPlayer){
      double theXAndPX = (playerLoc.get(i).x + theX)/2;
      double theYAndPY = (playerLoc.get(i).x + theY)/2;
      Point2D.Double point = new Point2D.Double(theXAndPX, theYAndPY);
      midPoints.add(point);
    }
  }
  //We want to add the distance for the p_1 to midpoint1 and the distance for p_2 to midpoint2
  //We want to add the distance for the P_1 to midpoint2 and the distance for p_2 to midpoint1
  //Then we take the action with the smallest sum.
  Player p1 = getPlayer(0);
  Player p2 = getPlayer(1);

  //ArrayList<Double> distSum = new ArrayList<Double>(numPlayers);
  //Sum the distances between the midpoints and our players, then use the smallest combination
  Point2D.Double midPoint = midPoints.get(0);
  double sum1 = distance(p1.x, p1.y, goalMidPoint.x, goalMidPoint.y) + distance(p2.x, p2.y, midPoint.x, midPoint.y);
  double sum2 = distance(p2.x, p2.y, goalMidPoint.x, goalMidPoint.y) + distance(p1.x, p1.y, midPoint.x, midPoint.y);
  if(sum1 < sum2){

    moveTo(p1, goalMidPoint.x, goalMidPoint.y);
    p1BlockingPosition = new Point2D.Double(goalMidPoint.x, goalMidPoint.y);
    moveTo(p2, midPoint.x, midPoint.y);
    p2BlockingPosition = new Point2D.Double(midPoint.x, midPoint.y);
  }else{
    moveTo(p2, goalMidPoint.x, goalMidPoint.y);
    p2BlockingPosition = new Point2D.Double(goalMidPoint.x, goalMidPoint.y);
    moveTo(p1, midPoint.x, midPoint.y);
    p1BlockingPosition = new Point2D.Double(midPoint.x, midPoint.y);
  }
}
}
