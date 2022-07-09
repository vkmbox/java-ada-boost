package vkmbox.classifier.util;

import java.util.List;

/**
 *
 * @author vkmbox
 */
public class GeometryUtil {
    
    public static double getNorm1(List<Double> list) {
        return list.stream().mapToDouble(Double::doubleValue).sum();
    }
}
