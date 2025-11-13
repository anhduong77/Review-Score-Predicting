package Classification_Algorithms;

import preprocess.Preprocessor;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Random;

public class Classification {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/data.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        FilteredClassifier fc_j48 = createFilteredClassifier( new J48());
        FilteredClassifier fc_nb = createFilteredClassifier ( new NaiveBayes());
        FilteredClassifier fc_rf =createFilteredClassifier  (new RandomForest());
        FilteredClassifier fc_log =createFilteredClassifier  (new SimpleLogistic());

    //FilteredClassifier fc_smo =createFilteredClassifier  (new SMO());

        Classifier[] classifiers = {fc_j48, fc_nb, fc_rf,fc_log};
        String[] names = { "J48 Decision Tree", "NaiveBayes", "RandomForest","Logistic Regression" };
        StringBuilder table = new StringBuilder();
        String line = "=".repeat(140);
        table.append("\n").append(line).append("\n");
        table.append(String.format("%-20s │ %-10s │ %-10s │ %-10s │ %-10s │ %-10s │ %-10s │ %-8s │ %-8s\n",
                "Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Kappa", "MAE", "RMSE"));
        table.append(line).append("\n");

        for (int i = 0; i < classifiers.length; i++) {
            try {
                System.out.println("\nProcessing: " + names[i] + "...");
                Evaluation eval = evaluateModel(classifiers[i], data);

                double accuracy = eval.pctCorrect();
                double precision = eval.weightedPrecision();
                double recall = eval.weightedRecall();
                double f1 = eval.weightedFMeasure();
                double auc = eval.weightedAreaUnderROC();
                double kappa = eval.kappa();

                double mae = eval.meanAbsoluteError();
                double rmse = eval.rootMeanSquaredError();


                table.append(String.format("%-20s │ %9.4f%% │ %10.4f │ %10.4f │ %10.4f │ %10.4f │ %10.4f │ %8.4f │ %8.4f\n",
                        names[i],
                        accuracy,
                        precision,
                        recall,
                        f1,
                        auc,
                        kappa,
                        mae,
                        rmse));
                System.out.println(eval.toMatrixString("\n Confusion Matrix (" + names[i] + ")"));
            }catch (Exception e){
                System.out.println("!!! Error " + names[i] + ": " + e.getMessage());
                table.append(String.format("%-17s | %s\n", names[i], "Error"));

        }

    }
        System.out.println(table.toString());
        System.out.println("=".repeat(80));




}



private static FilteredClassifier createFilteredClassifier(Classifier baseClassifier) throws Exception {
    FilteredClassifier fc = new FilteredClassifier();
    fc.setFilter(Preprocessor.createPreprocessor());
    fc.setClassifier(baseClassifier);
    return fc;
}
private static Evaluation evaluateModel(Classifier classifier, Instances data) throws Exception {
    Evaluation eval = new Evaluation(data);
    int numFolds = 10;
    Random rand = new Random(1);

    Instances randData = new Instances(data);
    randData.randomize(rand);

    eval.crossValidateModel(classifier, data, numFolds, rand);
    return eval;

}
}
