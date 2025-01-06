package clustering;

public class Kmeans {  

    int numClusters;
    Cluster[] clusters;
    Point[] dataset;
    int iterations;

    public Kmeans(int numClusters, Point[] dataset) {
        this.numClusters = numClusters;
        this.clusters = new Cluster[this.numClusters];
        this.dataset = dataset;
        this.iterations = 0;
    }

    public Cluster[] runKMeans() {
        createAndInitializeClusters();
        boolean check = true;
        Point[] oldCentroids = new Point[this.numClusters];
        
        do {
            this.iterations++;

            if (iterations > 1) {
                for (int i = 0; i < this.numClusters; i++) {
                    oldCentroids[i] = this.clusters[i].centroid;
                }
            }

            for (Cluster c : this.clusters) {
                c.points.clear();
            }

            for (Point p: this.dataset) {
                assignPointToCluster(p);
            }

            for (Cluster c: this.clusters) {
                calculateNewCentroid(c);
            }

            check = checkForTermination(this.clusters, oldCentroids);
        } while (check);


        printStats();
        return clusters;
    }
    
    private void createAndInitializeClusters() {
        for (int i = 0; i < this.numClusters; i++) {
            this.clusters[i] = new Cluster(i, chooseRandomPointAsCentroid());
        }
    }

    private Point chooseRandomPointAsCentroid() {
        return this.dataset[(int)(Math.random()*this.dataset.length)];
    }

    private void assignPointToCluster(Point p) {
        int cluster = findBestCluster(p);
        this.clusters[cluster].points.add(p);
    }

    private int findBestCluster(Point p) {
        double minDistance = Double.MAX_VALUE;
        double dist;
        int bestClusterId = -1;
        for (int i = 0; i < this.numClusters; i++) {
            dist = calculateEuclideanDistance(p, this.clusters[i].centroid);
            if (dist < minDistance) {
                bestClusterId = i;
                minDistance = dist;
            }
        }
        return bestClusterId;
    }


    private double calculateEuclideanDistance(Point p, Point c) {
        return Math.sqrt(Math.pow((c.x1 - p.x1), 2) + Math.pow((c.x2 - p.x2), 2));
    }

    private void calculateNewCentroid(Cluster c) {
        double sumX1 = 0;
        double sumX2 = 0;
        for (Point p: c.points) {
            sumX1 += p.x1;
            sumX2 += p.x2;
        }
        Point newCentroid = new Point(sumX1/c.points.size(), sumX2/c.points.size());
        c.centroid = newCentroid;
    } 

   
    private boolean checkForTermination(Cluster[] newCentroids, Point[] oldCentroids) {
        if (this.iterations == 1) {
            return true;
        }

        Point[] centroids = new Point[this.numClusters];
        for (int i = 0; i < this.numClusters; i++) {
            centroids[i] = newCentroids[i].centroid;
            if (calculateEuclideanDistance(centroids[i], oldCentroids[i]) < 0.0001) {
                return false;
            }
        }
        return true;
    }

    private void printStats() {
        System.out.println("KMeans terminated");
        System.out.println("Total iterations: " + this.iterations);
        for (Cluster c: this.clusters) {
            System.out.println("======================================================");
            System.out.println("Cluster "+c.clusterId+" centroid: ("+c.centroid.x1 +","+c.centroid.x2+")");
            System.out.println("Cluster "+c.clusterId+" has "+c.points.size()+" points");
        }
        System.out.println("======================================================");
        System.out.println("Total variance: " + calculateTotalVariance(this.clusters));

    }

    public double calculateTotalVariance(Cluster[] clusters) {
        double variance = 0;
        for (Cluster c: clusters) {
            for (Point p: c.points) {
                variance += calculateEuclideanDistance(p, c.centroid);
            }
        }
        return variance;
    }
}