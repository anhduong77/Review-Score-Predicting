package Classification_Algorithms;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;
import java.io.File;

public class SaveLoadModel {
    public static void main(String[] args) {
        final String DATA_FILE = "data/data.arff";      // file dữ liệu
        final String MODEL_FILE = "rf_model.model";     // file lưu mô hình

        try {
            // 1. Load dữ liệu
            System.out.println("1. Loading data from: " + DATA_FILE);
            DataSource source = new DataSource(DATA_FILE);
            Instances data = source.getDataSet();

            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            System.out.println("Data loaded successfully. Number of instances: " + data.numInstances());

            // 2. Loại bỏ thuộc tính date nếu có
            RemoveType removeDate = new RemoveType();
            removeDate.setOptions(new String[]{"-T", "date"});
            removeDate.setInputFormat(data);
            Instances filteredData = Filter.useFilter(data, removeDate);
            System.out.println("Date attributes removed. Number of attributes now: " + filteredData.numAttributes());

            // 3. Huấn luyện RandomForest
            System.out.println("\n2. Training RandomForest classifier...");
            RandomForest rf = new RandomForest();
            rf.setNumIterations(20);  // số cây, có thể thay đổi
            rf.buildClassifier(filteredData);
            System.out.println("RandomForest training completed.");

            // 4. Lưu mô hình
            System.out.println("\n3. Saving trained model to: " + MODEL_FILE);
            SerializationHelper.write(MODEL_FILE, rf);
            System.out.println("RandomForest model saved successfully.");

            // 5. Tải lại mô hình
            System.out.println("\n4. Loading model from: " + MODEL_FILE);
            File modelFile = new File(MODEL_FILE);
            if (!modelFile.exists()) {
                throw new Exception("Model file not found at: " + MODEL_FILE);
            }
            Classifier loadedModel = (Classifier) SerializationHelper.read(MODEL_FILE);
            System.out.println("RandomForest model loaded successfully!");

            // 6. Kiểm tra mô hình trên dữ liệu train
            System.out.println("\n5. Testing loaded model on the training data...");
            int correct = 0;
            for (int i = 0; i < filteredData.numInstances(); i++) {
                double actual = filteredData.instance(i).classValue();
                double pred = loadedModel.classifyInstance(filteredData.instance(i));
                if (actual == pred) correct++;
            }
            double accuracy = (100.0 * correct / filteredData.numInstances());
            System.out.println("\n--- Evaluation Results ---");
            System.out.println("Total instances: " + filteredData.numInstances());
            System.out.println("Correctly classified: " + correct);
            System.out.printf("Accuracy using loaded model: %.2f%%\n", accuracy);
            System.out.println("--------------------------");

        } catch (Exception e) {
            System.err.println("An error occurred during WEKA operation:");
            e.printStackTrace();
        }
    }
}
