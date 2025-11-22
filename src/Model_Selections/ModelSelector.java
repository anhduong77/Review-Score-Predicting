package Model_Selections;
import preprocess.Preprocessor;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Instance;
import java.util.Random;

public class ModelSelector {
    public static void selectBestModel(Instances data) throws Exception {
        FilteredClassifier fc_j48 = createFilterClassifier(new J48());
        FilteredClassifier fc_nb = createFilterClassifier(new NaiveBayes());
        FilteredClassifier fc_rf = createFilterClassifier(new RandomForest());
        FilteredClassifier fc_log = createFilterClassifier(new SimpleLogistic());

        Classifier[] models = {fc_j48, fc_nb, fc_rf, fc_log};
        String[] names = {"J48", "NaiveBayes", "RandomForest", "LogisticRegression"};

        double bestAUC = -1;
        Classifier bestModel = null;
        String bestName = "";

        System.out.println("\n============ MODEL SELECTION (AUC) ============");

        for (int i = 0; i < models.length; i++) {
            System.out.println("\nEvaluating: " + names[i]);
            Evaluation eval = evaluate(models[i], data);
            double auc = eval.weightedAreaUnderROC();
            System.out.println("AUC = " + auc);

            if (auc > bestAUC) {
                bestAUC = auc;
                bestModel = models[i];
                bestName = names[i];
            }
        }

        System.out.println("\nBest Model = " + bestName + "(AUC = " + bestAUC + ")");

        // Training the best model
        bestModel.buildClassifier(data);

        // Finding threshold
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(bestModel, data, 10, new Random(1));

        double bestThreshold = ThresholdHelper.findBestThreshold(eval, 1);
        System.out.println(">>> Best Threshold = " + bestThreshold);

        System.out.println("\n=========== FINAL PREDICTION (with threshold) ============");
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double[] probs = bestModel.distributionForInstance(inst);
            int pred = ThresholdHelper.predictWithThreshold(probs, bestThreshold);
            System.out.println(("Row " + i + " -> Pred = " + pred + " | Prob = " + probs[1]));
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.printf(">>> ALL DONE! Total instances processed: %,d%n", data.numInstances());
        System.out.printf(">>> BEST MODEL           : %s%n", bestName);
        System.out.printf(">>> BEST WEIGHTED AUC    : %.4f%n", bestAUC);
        System.out.printf(">>> BEST THRESHOLD (F1)  : %.6f%n", bestThreshold);
        System.out.println("=".repeat(80));
    }

    private static FilteredClassifier createFilterClassifier(Classifier classifier) {
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(Preprocessor.createPreprocessor());
        fc.setClassifier(classifier);
        return fc;
    }

    private static Evaluation evaluate(Classifier classifier, Instances data) throws Exception{
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        return eval;
    }
}
