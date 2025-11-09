import preprocess.Preprocessor;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;


public class App {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data/data.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(Preprocessor.createPreprocessor());
        fc.setClassifier(new J48());
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(fc, data, 10, new Random(1));
        System.out.println(eval.toSummaryString());
    }
}
