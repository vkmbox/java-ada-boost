package vkmbox.classifier.ensemble;

import static org.nd4j.common.util.MathUtils.sum;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.RequiredArgsConstructor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import vkmbox.classifier.util.RademacherUtil;
import vkmbox.classifier.base.DecisionStumpContinuous;

import java.util.Map;
import java.util.List;
import java.time.Clock;
import java.time.ZoneId;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;
import java.util.ArrayList;

@Slf4j
public class AdaBoostStandardClassifier {
    
    private static final int[] SIGNS = {-1, 1};
    private static final Clock CLOCK = Clock.system(ZoneId.systemDefault());
    
    private int estimators;
    private final double tolerance;
    private final List<Double> ensembleAlphas = new ArrayList<>();
    private final List<DecisionStumpContinuous> ensembleClassifiers = new ArrayList<>();
    private double alphaNorm1;
    
    public AdaBoostStandardClassifier(int estimators, double tolerance) {
        this.estimators = estimators;
        this.tolerance = tolerance;
    }

    public AdaBoostStandardClassifier(int estimators) {
        this(estimators, 1e-10);
    }
    
    public FitResult fitIND(INDArray dataX, INDArray dataY, double[] sampleWeights, boolean trace) {
        ensembleAlphas.clear();
        ensembleClassifiers.clear();
        long timeStart = CLOCK.millis();
        int samplesCount = dataX.rows();
        double[] weights = sampleWeights;
        if (Objects.isNull(sampleWeights)) {
            weights = new double[samplesCount];
            Arrays.fill(weights, 1.0/samplesCount);
        }
        Map<String, List> history = trace? new HashMap<>(): null;

        for (int dummy = 0; dummy < estimators; dummy++) {
            //INDArray weightsMulti = Nd4j.tile(weights, featuresCount); //(1, features_count)); //dd_t
            var optimum = getDecisionStump(dataX, dataY, weights);
            if (optimum.minimalError >= 0.5) {
                return FitResult.of("error_level_exceeded", history);
            }
            double alphaT = 0.5 * Math.log((1 - optimum.minimalError)/(optimum.minimalError+tolerance));
            ensembleAlphas.add(alphaT);
            ensembleClassifiers.add(optimum.decisionStump);
            log(trace, history, optimum.minimalError, weights, timeStart);
            if (optimum.minimalError == 0) {
                alphaNorm1 = ensembleAlphas.stream().mapToDouble(Double::doubleValue).sum() + tolerance;
                return FitResult.of("error_free_classifier_found", history);
            }
            int[] forecast = optimum.decisionStump.classify(dataX);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                weights[sampleNumber] *= Math.exp(-alphaT * dataY.getInt(sampleNumber) * forecast[sampleNumber]);
            }
            double weightSum = sum(weights);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                weights[sampleNumber] = weights[sampleNumber]/(weightSum+tolerance);
            }
        }
        alphaNorm1 = ensembleAlphas.stream().mapToDouble(Double::doubleValue).sum() + tolerance;
        return FitResult.of("iterations_exceeded", history);
    }
    
    public String fitIND(INDArray dataX, INDArray dataY) {
        var response = fitIND(dataX, dataY, null, false);
        return response.result;
    }
    
    public FitResult fitIND(INDArray dataX, INDArray dataY, boolean trace) {
        return fitIND(dataX, dataY, null, trace);
    }
    
    public String fit(double[][] dataX, int[] dataY, int estimators) {
        this.estimators = estimators;
        return fit(dataX, dataY);
    }
    
    public String fit(double[][] dataX, int[] dataY) {
        log.info("fit is called. Data: {} rows, {} columns", dataX.length, dataX[0].length);
        INDArray indY = Nd4j.create(dataY, new long[]{dataY.length}, DataType.INT32);
        var response = fitIND(Nd4j.create(dataX), indY, null, false);
        return response.result;
    }
    
    public INDArray predictRaw(INDArray dataX) {
        int samplesCount = dataX.rows();
        INDArray buffer = Nd4j.zeros(new int[]{samplesCount});
        for (int index = 0; index < ensembleAlphas.size(); index++) {
            buffer.addi(ensembleClassifiers.get(index)
                .classifyIND(dataX, DataType.DOUBLE).mul(ensembleAlphas.get(index)));
        }
        return buffer.divi(alphaNorm1);
    }

    public INDArray predictIND(INDArray dataX) {
        return Transforms.sign(predictRaw(dataX));
    }
    
    public int[] predict(double[][] dataX) {
        return predictIND(Nd4j.create(dataX)).toIntVector();
    }

    public double getMarginL1(double[][] dataX) {
        return getMarginL1IND(Nd4j.create(dataX));
    }
    
    public double getMarginL1IND(INDArray dataX) {
        INDArray buffer = predictRaw(dataX);
        //double alphaModulo = ensembleAlphas.stream().mapToDouble(Double::doubleValue).sum() + tolerance;
        return buffer.aminNumber().doubleValue();//alphaModulo;
    }
    
    public double calculateRademacherForBiclassifiers(double[][] dataX, long subsetSize) {
        return calculateRademacherForBiclassifiersIND(Nd4j.create(dataX), subsetSize);
    }
    
    public double calculateRademacherForBiclassifiersIND(INDArray dataX, long subsetSize) {
        return RademacherUtil.calculateForBiclassifiers(dataX, subsetSize);
    }
    
    public double calculateMarginLoss(double[][] dataX, int[] dataY, double rho) {
        INDArray indY = Nd4j.create(dataY, new long[]{dataY.length}, DataType.INT32);
        return calculateMarginLossIND(Nd4j.create(dataX), indY, rho);
    }
    
    public double calculateMarginLossIND(INDArray dataX, INDArray dataY, double rho) {
        int samplesCount = dataX.rows();
        INDArray indY = Nd4j.create(dataY.toDoubleVector());
        double[] buffer = indY.mul(predictRaw(dataX)).toDoubleVector();
        double sum = 0.0;
        for (double value: buffer) {
            if (value <= 0) {
                sum += 1.0;
            } else if (value <= rho) {
                sum += 1.0 - (value/rho);
            }
        }
        return sum/samplesCount;
    }
    
    private StampResult getDecisionStump(INDArray dataX, INDArray dataY, double[] weights) {
        int samplesCount = dataX.rows(), featuresCount = dataX.columns();
        double minimalError = Double.MAX_VALUE, minimalThreshold = Double.MAX_VALUE;
        int minimalFeature = 0, minimalSign = 0;

        for (int featureNumber = 0; featureNumber < featuresCount; featureNumber++) {
            INDArray feature = dataX.getColumn(featureNumber);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                double threshold = feature.getDouble(sampleNumber);
                for (int sign : SIGNS) {
                    double currentError = DecisionStumpContinuous.getError
                        (dataX, dataY, featureNumber, weights, sign, threshold);
                    if (minimalError > currentError) {
                        minimalError = currentError;
                        minimalFeature = featureNumber;
                        minimalSign = sign; 
                        minimalThreshold = threshold;
                    }
                }
            }
        }
        return StampResult.of(minimalError
                , new DecisionStumpContinuous(minimalFeature, minimalSign, minimalThreshold));
    }

    private void log(boolean trace, Map<String, List> history
            , double minimalError, double[] weights, long timeStart) {
        if (trace) {
            history.computeIfAbsent("time", arg -> new ArrayList<Long>()).add(CLOCK.millis() - timeStart);
            history.computeIfAbsent("error", arg -> new ArrayList<Double>()).add(minimalError);
            history.computeIfAbsent("d_t", arg -> new ArrayList<double[]>()).add(weights);
        }
    }

    @Getter
    @RequiredArgsConstructor
    public static class StampResult {
        private final Double minimalError;
        private final DecisionStumpContinuous decisionStump;
        
        public static StampResult of(Double minimalError, DecisionStumpContinuous decisionStump) {
            return new StampResult(minimalError, decisionStump);
        }
    }
    
    @Getter
    @RequiredArgsConstructor
    public static class FitResult {
        private final String result;
        private final Map<String, List> history;
        
        public static FitResult of(String result, Map<String, List> history) {
            return new FitResult(result, history);
        }
    }

}
