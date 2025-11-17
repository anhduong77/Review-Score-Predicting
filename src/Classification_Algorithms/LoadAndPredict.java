package Classification_Algorithms;

import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class LoadAndPredict {
    public static void main(String[] args) {
        final String MODEL_FILE = "rf_model.model";
        final String TEMPLATE_FILE = "data/filtered_template.arff"; // template filteredData

        try {
            // 1. Load model đã train
            Classifier loadedModel = (Classifier) SerializationHelper.read(MODEL_FILE);
            System.out.println("Model loaded successfully!");

            // 2. Load template filteredData để giữ cấu trúc attribute
            DataSource source = new DataSource(TEMPLATE_FILE);
            Instances newData = source.getDataSet();
            newData.setClassIndex(newData.numAttributes() - 1);
            System.out.println("Template data loaded. Num attributes: " + newData.numAttributes());

            // 3. Tạo một instance mới
            double[] vals = new double[newData.numAttributes()];

            // --- Set giá trị cho các attribute theo index ---
            // Ví dụ: bạn gán giá trị 100 cho tất cả numeric, first nominal index = 0
            for (int i = 0; i < newData.numAttributes() - 1; i++) { // trừ class
                if (newData.attribute(i).isNumeric()) {
                    vals[i] = 100.0; // ví dụ numeric
                } else if (newData.attribute(i).isNominal()) {
                    vals[i] = 0; // index đầu tiên
                }
            }

            // Class attribute chưa biết → NaN
            vals[newData.classIndex()] = Double.NaN;

            // Thêm instance vào newData
            newData.add(new DenseInstance(1.0, vals));

            // 4. Dự đoán
            System.out.println("\nPredictions:");
            for (int i = 0; i < newData.numInstances(); i++) {
                double pred = loadedModel.classifyInstance(newData.instance(i));
                String predictedClass = newData.classAttribute().value((int) pred);
                System.out.println("Instance " + i + " => Prediction: " + predictedClass);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
