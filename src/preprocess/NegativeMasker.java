// package filter;

package preprocess;

import weka.core.*;
import java.util.*;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;

public class NegativeMasker extends SimpleBatchFilter implements UnsupervisedFilter {

    private String attributeName;
    private double threshold = 0.0;

    public void setAttributeName(String attributeName) {
        this.attributeName = attributeName;
    }

    public String getAttributeName() {
        return this.attributeName;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public double getThreshold() {
        return this.threshold;
    }

    @Override
    public Instances determineOutputFormat(Instances instanceInfo) throws Exception {
        

        ArrayList<String> maskValues = new ArrayList<>();
        maskValues.add("No");
        maskValues.add("Yes");
        Attribute maskAttr = new Attribute(attributeName + "_mask", maskValues);

        Instances newFormat = new Instances(instanceInfo);
        int insertPos = (instanceInfo.classIndex() >= 0)
                ? instanceInfo.classIndex()
                : instanceInfo.numAttributes();
        newFormat.insertAttributeAt(maskAttr, insertPos);

        
        return newFormat;
    }

  
    @Override
    protected Instances process(Instances data) throws Exception {  
        Instances newData = new Instances(determineOutputFormat(data), data.numInstances());
        Attribute targetAttr = data.attribute(attributeName);
        if (targetAttr == null || !targetAttr.isNumeric()) {
            throw new IllegalArgumentException("Attribute " + attributeName + " not found or not numeric.");
        }

  
        int maskIndex = newData.attribute(attributeName + "_mask").index();

        for (int i = 0; i < data.numInstances(); i++) {
            Instance oldInst = data.instance(i);
            double[] newValues = new double[newData.numAttributes()];

            for (int j = 0; j < data.numAttributes(); j++) {
                int newPos = (j < maskIndex) ? j : j + 1;
                newValues[newPos] = oldInst.value(j);
            }

            if (oldInst.isMissing(targetAttr)) {
                newValues[maskIndex] = Utils.missingValue();
            } else {
                String mask = (oldInst.value(targetAttr) < threshold) ? "Yes" : "No";
                newValues[maskIndex] = newData.attribute(maskIndex).indexOfValue(mask);
            }

            Instance newInst = new DenseInstance(oldInst.weight(), newValues);
            newData.add(newInst);
        }

        return newData;
    }

    @Override
    protected boolean hasImmediateOutputFormat() {
        return true;
    }

    @Override 
    public boolean batchFinished() throws Exception {
        return super.batchFinished();
    }
    @Override 
    public String globalInfo() {
        return "A filter that binarize value by a specific threshold.";
    }

}

