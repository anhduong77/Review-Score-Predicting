package preprocess;


import weka.core.*;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;
import java.util.*;

public class PercentileClip extends SimpleBatchFilter implements UnsupervisedFilter {

    private double lowerPercentile = 2.0;
    private double upperPercentile = 98.0;
    private String[] attributes;

    public void setLowerPercentile(double p) {
        this.lowerPercentile = p;
    }

    public void setUpperPercentile(double p) {
        this.upperPercentile = p;
    }

    public double getLowerPercentile() {
        return lowerPercentile;
    }

    public double getUpperPercentile() {
        return upperPercentile;
    }
    public void setAttributes(String attrNames) {
        this.attributes = attrNames.split(",");
    }
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {

        return new Instances(inputFormat, 0);
    }

    @Override
    protected Instances process(Instances data) throws Exception {
        Instances output = new Instances(determineOutputFormat(data), data.numInstances());

        Map<Integer, double[]> bounds = new HashMap<>();

        for (String a: attributes) {
            a = a.trim();
            Attribute attr = data.attribute(a);
            int i = attr.index();
            if (attr.isNumeric() && i != data.classIndex()) {
                List<Double> values = new ArrayList<>();
                for (int j = 0; j < data.numInstances(); j++) {
                    Instance inst = data.instance(j);
                    if (!inst.isMissing(attr))
                        values.add(inst.value(attr));
                }

                if (values.size() > 0) {
                    Collections.sort(values);
                    double lower = percentile(values, lowerPercentile);
                    double upper = percentile(values, upperPercentile);
                    bounds.put(i, new double[]{lower, upper});
                }
            }
        }

        for (int i = 0; i < data.numInstances(); i++) {
            Instance oldInst = data.instance(i);
            double[] vals = oldInst.toDoubleArray();

            for (Map.Entry<Integer, double[]> entry : bounds.entrySet()) {
                int index = entry.getKey();
                double lower = entry.getValue()[0];
                double upper = entry.getValue()[1];
                if (!Double.isNaN(vals[index])) {
                    vals[index] = Math.max(lower, Math.min(vals[index], upper));
                }
            }

            Instance newInst = new DenseInstance(oldInst.weight(), vals);
            output.add(newInst);
        }

        return output;
    }

    private double percentile(List<Double> sortedValues, double p) {
        if (sortedValues.isEmpty()) return Double.NaN;
        if (p <= 0) return sortedValues.get(0);
        if (p >= 100) return sortedValues.get(sortedValues.size() - 1);
        double rank = (p / 100.0) * (sortedValues.size() - 1);
        int low = (int) Math.floor(rank);
        int high = (int) Math.ceil(rank);
        if (low == high) return sortedValues.get(low);
        double weight = rank - low;
        return sortedValues.get(low) * (1 - weight) + sortedValues.get(high) * weight;
    }

    @Override
    protected boolean hasImmediateOutputFormat() {
        return true;
    }

    @Override
    public String globalInfo() {
        return "A filter that clips numeric attributes to the given percentile range (e.g., 2–98%).";
    }
}

