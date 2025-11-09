package preprocess;

import weka.core.*;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;
import java.util.*;

public class DateTimeSubtract extends SimpleBatchFilter implements UnsupervisedFilter {

    private final List<String[]> pairs = new ArrayList<>();

    public void addPair(String startAttr, String endAttr, String newAttrName) {
        pairs.add(new String[]{startAttr, endAttr, newAttrName});
    }
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        Instances newFormat = new Instances(inputFormat, 0);
        int insertPos = (inputFormat.classIndex() >= 0)
                ? inputFormat.classIndex()
                : inputFormat.numAttributes();

        for (String[] p : pairs) {
            Attribute diffAttr = new Attribute(p[2]);
            newFormat.insertAttributeAt(diffAttr, insertPos);
            insertPos++; 
        }

        return newFormat;
    }


    @Override
    protected Instances process(Instances data) throws Exception {
        Instances output = new Instances(determineOutputFormat(data), data.numInstances());

        for (int i = 0; i < data.numInstances(); i++) {
            Instance oldInst = data.instance(i);
            double[] oldVals = oldInst.toDoubleArray();
            double[] newVals = Arrays.copyOf(oldInst.toDoubleArray(), output.numAttributes());

            for (String[] p : pairs) {
     
                Attribute start = data.attribute(p[0]);
                Attribute end = data.attribute(p[1]);
                Attribute diffAttr = getOutputFormat().attribute(p[2]);
                int newIndex = diffAttr.index();
                newVals[newVals.length-1] = oldVals[oldVals.length-1];
                if (oldInst.isMissing(start) || oldInst.isMissing(end)) {
                    newVals[newIndex] = Utils.missingValue();
                } else {
                    double diffDays = (oldInst.value(end) - oldInst.value(start)) / (1000 * 60 * 60 * 24);
                    newVals[newIndex] = diffDays;
                }
            }

            Instance newInst = new DenseInstance(oldInst.weight(), newVals);
            output.add(newInst);
        }

        return output;
    }

    @Override
    protected boolean hasImmediateOutputFormat() {
        return true;
    }

    @Override 
    public String globalInfo() {
        return "A filter that subtracts date/time attributes to create new numeric attributes representing the difference in days.";
    }

}

  


