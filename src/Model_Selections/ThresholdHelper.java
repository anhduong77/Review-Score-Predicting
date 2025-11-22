package Model_Selections;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.evaluation.ThresholdCurve;

public class ThresholdHelper {
    public static double findBestThreshold(Evaluation eval, int pos) {
        ThresholdCurve tc = new ThresholdCurve();
        Instances curve = tc.getCurve(eval.predictions());

        double bestF1 = -1;
        double bestT = 0.5;

        for (int i = 0; i < curve.numInstances(); i++) {
            Instance point = curve.instance(i);

            double precision = point.value(curve.attribute(ThresholdCurve.PRECISION_NAME));
            double recall = point.value(curve.attribute(ThresholdCurve.RECALL_NAME));
            double thres = point.value(curve.attribute((ThresholdCurve.THRESHOLD_NAME)));
            double f1 = 2 * precision * recall / (precision + recall + 1e-10);

            if (f1 > bestF1) {
                bestF1 = f1;
                bestT = thres;
            }
        }

        System.out.println("Best threshold found: " + bestT + " with F1 = " + bestF1);
        return bestT;
    }

    public static int predictWithThreshold(double[] probs, double threshold) {
        if (probs == null || probs.length < 2) {
            throw new IllegalArgumentException("Probs array must have at least 2 elements");
        }

        return probs[1] >= threshold ? 1:0;
    }
}
