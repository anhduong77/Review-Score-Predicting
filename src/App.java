import preprocess.*;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;

import weka.classifiers.Classifier;
import java.util.Random;





public class App {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data/data.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(Preprocessor.createPreprocessor());

    }
}