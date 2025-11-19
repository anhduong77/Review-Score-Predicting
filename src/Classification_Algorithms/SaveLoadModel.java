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


    static final String TRAIN_FILE = "data/data.arff";
    static final String TEST_FILE = "data/test_data.arff";
    static final String MODEL_FILE = "rf_model.model";

    public static void main(String[] args) {
        try {

            System.out.println("=== Starting Traning ===");


            DataSource source = new DataSource(TRAIN_FILE);
            Instances trainData = source.getDataSet();
            if (trainData.classIndex() == -1) trainData.setClassIndex(trainData.numAttributes() - 1);


            RemoveType removeDate = new RemoveType();
            removeDate.setOptions(new String[]{"-T", "date"});
            removeDate.setInputFormat(trainData);
            Instances processedTrainData = Filter.useFilter(trainData, removeDate);

            // Train Model
            RandomForest rf = new RandomForest();
            rf.buildClassifier(processedTrainData);

            // Save Model (SerializationHelper.write)
            SerializationHelper.write(MODEL_FILE, rf);
            System.out.println("-> Đã lưu mô hình vào: " + MODEL_FILE);



            System.out.println("\n=== (LOAD & PREDICT) ===");


            Classifier loadedModel = (Classifier) SerializationHelper.read(MODEL_FILE);
            System.out.println("->Load successful");


            File testFileCheck = new File(TEST_FILE);
            if (testFileCheck.exists()) {
                DataSource testSource = new DataSource(TEST_FILE);
                Instances testData = testSource.getDataSet();
                if (testData.classIndex() == -1) testData.setClassIndex(testData.numAttributes() - 1);


                Instances processedTestData = Filter.useFilter(testData, removeDate);

                System.out.println("\n---Predict Result ---");
                for (int i = 0; i < processedTestData.numInstances(); i++) {
                    double pred = loadedModel.classifyInstance(processedTestData.instance(i));
                    String predLabel = processedTestData.classAttribute().value((int) pred);

                    System.out.println("numInstance " + (i+1) + ": Predict = " + predLabel);
                }
            } else {
                System.out.println("Warning:Do not find Test file (" + TEST_FILE + ").");

            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}