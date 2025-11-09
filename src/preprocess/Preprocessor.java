package preprocess;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import preprocess.PercentileClip;
public class Preprocessor {

    public static MultiFilter createPreprocessor() {
        MultiFilter pipeline = new MultiFilter();
        String[][] configs = {
            {"order_purchase_timestamp", "order_approved_at", "approved_purchase"},
            {"order_purchase_timestamp", "order_delivered_carrier_date", "carried_purchase"},
            {"order_purchase_timestamp", "order_delivered_customer_date", "delivered_purchase"},
            {"order_purchase_timestamp", "order_estimated_delivery_date", "estimated_purchase"},
            {"order_purchase_timestamp", "shipping_limit_date", "shipping_limit_purchase"},
            {"order_estimated_delivery_date", "order_delivered_customer_date", "estimated_error"}
        };
        DateTimeSubtract DateTimeFilter = new DateTimeSubtract();
        for (String[] conf : configs) {
            DateTimeFilter.addPair(conf[0], conf[1], conf[2]);
        }
        
        String regex = "(estimated_error|delivered_purchase|estimated_error_mask|order_status|carried_purchase|shipping_limit_purchase|seller_zip_code_prefix|freight_value|product_width_cm|product_height_cm|product_length_cm|product_weight_cm|price|approved_purchase|review_score)";
        RemoveByName removeByNameFilter = new RemoveByName();
        removeByNameFilter.setExpression(regex);
        removeByNameFilter.setInvertSelection(true);


        NegativeMasker negativeMasker = new NegativeMasker();
        negativeMasker.setAttributeName("estimated_error");
        
        ReplaceMissingValues missingFiller = new ReplaceMissingValues();


        Normalize normalize = new Normalize();
        normalize.setIgnoreClass(true);  

        PercentileClip percentileClip = new PercentileClip();
        percentileClip.setAttributes("approved_purchase,carried_purchase,delivered_purchase");
       
      

        pipeline.setFilters(new Filter[] {
            DateTimeFilter,
            negativeMasker,
            removeByNameFilter,
            percentileClip,
            missingFiller,
            normalize
        });

        return pipeline;
    }
}
