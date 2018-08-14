import java.util.*;
import java.io.*;
import org.apache.commons.lang3.builder.HashCodeBuilder;

public class RLState implements Serializable {

    // -1 for opponent scoring; 0 for no scoring; 1 for us scoring
    int ballState;

    boolean teamHasBall;

    // integer array of size 3 with conditionally values based on teamHasBall
    // teamHasBall == true  --> -1 is not open, 0 is has ball, 1 is open
    // teamHasBall == false --> -1 is not blocking a pass, 0 is exclusively used
    //                          for the goal player, 1 is blocking a pass
    // index 0 = player0; index 1 = player1; index2 = goal
    int[] teamStatus;

    // team locations according to a 3x3 grid of the soccer field
    // index 0 = player0; index 1 = player1;
    int[] teamLocations;

    // opp locations according to a 3x3 grid of the soccer field
    // index 0 = player0; index 1 = player1;
    int[] oppLocations;

    public RLState() {
        ballState = 0;
        teamHasBall = false;
        teamStatus = new int[3];
        teamLocations = new int[2];
        oppLocations = new int[2];
    }

    public RLState(int ballState, boolean teamHasBall, int[] teamStatus, int[] teamLocations, int[] oppLocations) {
        this.ballState = ballState;
        this.teamHasBall = teamHasBall;
        this.teamStatus = teamStatus;
        this.teamLocations = teamLocations;
        this.oppLocations = oppLocations;
    }

    public String toString() {
        return ballState +
               " " + teamHasBall +
               " " + Arrays.toString(teamStatus) +
               " " + Arrays.toString(teamLocations) +
               " " + Arrays.toString(oppLocations);
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof RLState) {
            RLState rls = (RLState) o;
            return (ballState == rls.ballState && teamHasBall == rls.teamHasBall &&
                Arrays.equals(teamStatus, rls.teamStatus) &&
                Arrays.equals(teamLocations, rls.teamLocations) &&
                Arrays.equals(oppLocations, rls.oppLocations));
        }
        return false;
    }

    @Override
    public int hashCode() {
    	return new HashCodeBuilder()
    			.append(ballState)
    			.append(teamHasBall)
    			.append(teamStatus)
    			.append(teamLocations)
    			.append(oppLocations)
    			.build();
    }
}
