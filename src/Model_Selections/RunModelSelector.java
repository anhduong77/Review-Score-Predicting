package Model_Selections;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class RunModelSelector
{
    public static void main(String[] args) throws Exception {
        // Load Data
        ConverterUtils.DataSource src = new ConverterUtils.DataSource("data/data.arff");
        Instances data = src.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Run Model + threshold
        ModelSelector.selectBestModel(data);
    }
}
