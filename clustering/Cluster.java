package clustering;

import java.util.ArrayList;
import java.util.List;

public class Cluster {

    List<Point> points;
    int clusterId;
    Point centroid;

    public Cluster(int clusterId, Point centroid) {
        this.points = new ArrayList<>();
        this.clusterId = clusterId;
        this.centroid = centroid;
    }
}
