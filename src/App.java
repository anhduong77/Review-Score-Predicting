import preprocess.Preprocessor;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class App {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data/data.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(Preprocessor.createPreprocessor());
       
    }
}
