package Classification_Algorithms;

import preprocess.Preprocessor;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class CombineModel {
    public static void main(String[] args) throws Exception {
        // Load dữ liệu
        DataSource source = new DataSource("data/data.arff"); // kiểm tra đường dẫn
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // ---------- AdaBoostM1 ----------
        AdaBoostM1 adaBoost = new AdaBoostM1();
        adaBoost.setClassifier(new J48()); // Base learner: J48
        adaBoost.setNumIterations(50); // số vòng boosting

        FilteredClassifier fcAda = new FilteredClassifier();
        fcAda.setClassifier(adaBoost);
        fcAda.setFilter(Preprocessor.createPreprocessor());

        Evaluation evalAda = new Evaluation(data);
        evalAda.crossValidateModel(fcAda, data, 10, new Random(1));

        System.out.println("=== AdaBoostM1 (with J48) ===");
        System.out.println("Accuracy: " + (1 - evalAda.errorRate()) * 100 + "%");
        System.out.println(evalAda.toSummaryString());
        System.out.println(evalAda.toClassDetailsString());
        System.out.println(evalAda.toMatrixString());

        Bagging bagging = new Bagging();
        bagging.setClassifier(new J48()); // Base learner: J48
        bagging.setNumIterations(50); // số cây

        FilteredClassifier fcBag = new FilteredClassifier();
        fcBag.setClassifier(bagging);
        fcBag.setFilter(Preprocessor.createPreprocessor());

        Evaluation evalBag = new Evaluation(data);
        evalBag.crossValidateModel(fcBag, data, 10, new Random(1));

        System.out.println("=== Bagging (with J48) ===");
        System.out.println("Accuracy: " + (1 - evalBag.errorRate()) * 100 + "%");
        System.out.println(evalBag.toSummaryString());
        System.out.println(evalBag.toClassDetailsString());
        System.out.println(evalBag.toMatrixString());


        // ---------- Stacking ----------
        Stacking stack = new Stacking();

        // Base classifiers
        weka.classifiers.Classifier[] baseModels = {
                new RandomForest(),
                new J48(),
                new NaiveBayes()
        };
        stack.setClassifiers(baseModels);

        // Meta-classifier
        stack.setMetaClassifier(new Logistic());

        // FilteredClassifier giữ pipeline tiền xử lý
        FilteredClassifier fcStack = new FilteredClassifier();
        fcStack.setClassifier(stack);
        fcStack.setFilter(Preprocessor.createPreprocessor());

        // Evaluation bằng 10-fold cross-validation
        Evaluation evalStack = new Evaluation(data);
        evalStack.crossValidateModel(fcStack, data, 10, new Random(1));

        System.out.println("=== Stacking (RF + J48 + NaiveBayes, meta: Logistic) ===");
        System.out.println("Accuracy: " + (1 - evalStack.errorRate()) * 100 + "%");
        System.out.println(evalStack.toSummaryString());
        System.out.println(evalStack.toClassDetailsString());
        System.out.println(evalStack.toMatrixString());
    }
}


