import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;


public class Main {

    public static void main(String args[]) throws IOException {
        String testFileName = "/Users/riotriot/Desktop/KNN-ml2/test.txt";
        String trainFileName = "/Users/riotriot/Desktop/KNN-ml2/train.txt";

        File testFile = new File(testFileName);
        File trainFile = new File(trainFileName);

        List<Vector> testList = KNN.storeFile(testFile);
        List<Vector> trainList = KNN.storeFile(trainFile);

        System.out.println("Enter the k: ");
        Scanner scanner = new Scanner(System.in);
        int k = scanner.nextInt();
        System.out.println("The number of the nearest neighbors chosen:  " + k);

        KNN.calculateDistances(testList,trainList);
        KNN.KNNalgo(testList,k);

        System.out.println("______________________________");

        System.out.println("Plotting accuracy versus k...");
        for (int i = 1; i <=120 ; i++) {
            KNN.KNNalgo(testList,i);
            System.out.println();
        }

        System.out.println("______________________________");

        System.out.println("Enter the values of the vector\t the number of values need to be as regards training file");

        testList.clear();
        testList.add(KNN.scanVector());

        KNN.calculateDistances(testList,trainList);
        KNN.KNNalgo(testList,k);
    }
}

class Distance implements Comparable<Distance>{
    Vector from;
    Vector to;

    double distance;

    public Distance(Vector from, Vector to, double distance) {
        this.from = from;
        this.to = to;
        this.distance = distance;
    }

    public Vector getFrom() {
        return from;
    }

    public Vector getTo() {
        return to;
    }

    public double getDistance() {
        return distance;
    }

    @Override
    public String toString() {
        return to.getIrisClass() + " - " + from.getIrisClass() + ": " + distance;
    }

    @Override
    public int compareTo(Distance otherDistance) {
        return (Double.compare(this.getDistance(), otherDistance.getDistance()));
    }
}


class Vector {

    private double[] values;
    private String IrisClass;
    private List<Distance> distances;
    private String predictedClass;

    public Vector(double[] values) {
        this.values = values;
        IrisClass="";
        distances = new ArrayList<>();
    }

    public Vector(double[] values, String irisClass) {
        this.values = values;
        IrisClass = irisClass;
        distances = new ArrayList<>();
    }

    public List<Distance> getDistances() {
        return distances;
    }

    public double[] getValues() {
        return values;
    }

    public String getIrisClass() {
        return IrisClass;
    }

    public void addDistance(Distance distance){
        distances.add(distance);
        distances.sort(Distance::compareTo);
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public void setPredictedClass(String predictedClass) {
        this.predictedClass = predictedClass;
    }


    @Override
    public String toString() {
        return IrisClass + " " + Arrays.toString(values);
    }

}

class KNN {

    public static List<Vector> storeFile(File file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        String irisClass;
        List<Vector> vectors = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            int i;
            String[] lineSplit = line.split(",");
            double[] vectorValues = new double[lineSplit.length - 1];
            for (i = 0; i < vectorValues.length; i++) {
                vectorValues[i] = Double.parseDouble(lineSplit[i]);
            }
            irisClass = lineSplit[i];
            vectors.add(new Vector(vectorValues, irisClass));
        }
        br.close();
        return vectors;
    }

    public static void KNNalgo(List<Vector> test, int k) {
        int wrong = 0;
        for (Vector testVector : test) {
            testVector.setPredictedClass(mostFrequent(testVector.getDistances(), k));
            //   System.out.println("Original class is: " + testVector.getIrisClass() + ", tested one is: " + testVector.getPredictedClass()); //log
            if (!testVector.getIrisClass().isEmpty() && !testVector.getIrisClass().equals(testVector.getPredictedClass())) {
                wrong++;
                System.out.println("Original class is: " + testVector.getIrisClass() + ", predicted one is: " + testVector.getPredictedClass()); //log
            }
        }
        double res = 100 - (((double) wrong / (double) test.size()) * 100);
        System.out.println("Accuracy with " + k + " nearest neighbours is: " + res);
    }

    private static double getDistance(double[] vector1, double[] vector2) {
        double sum = 0;
        for (int i = 0; i < vector1.length; i++)
            sum += Math.pow(vector1[i] - vector2[i], 2);
        return Math.sqrt(sum);
    }

    private static String mostFrequent(List<Distance> list, int k) {
        return list.stream()
                .sorted()
                .limit(k)
                .map(Distance::getTo)
                .map(Vector::getIrisClass)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                .entrySet()
                .stream()
                .max(Comparator.comparing(Map.Entry::getValue))
                .get()
                .getKey()
                ;
    }

    public static Vector scanVector() {
        Scanner scanner = new Scanner(System.in);

        ArrayList<Double> inputVector = new ArrayList<>();

        while (scanner.hasNextDouble()) {
            inputVector.add(scanner.nextDouble());
        }
        double[] inputDoubles = new double[inputVector.size()];
        for (int i = 0; i < inputDoubles.length; i++) {
            inputDoubles[i] = inputVector.get(i);
        }
        return new Vector(inputDoubles);
    }

    public static void calculateDistances(List<Vector> test, List<Vector> train) {
        for (Vector testVector : test) {
            for (Vector trainVector : train) {
                testVector.addDistance(new Distance(testVector, trainVector,
                        getDistance(testVector.getValues(), trainVector.getValues())));
            }
        }
    }
}