package vkmbox.classifier.ensemble;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import vkmbox.classifier.adaboost.AdaBoostParallelClassifier;
import vkmbox.classifier.adaboost.AdaBoostStandardClassifier;

import java.time.Clock;
import java.time.ZoneId;
import java.util.Random;

@Slf4j
public class ResultsSimilarityTest {
    
    protected static final Clock CLOCK = Clock.system(ZoneId.systemDefault());

    @Test 
    void arraysCall() {
        final int SAMPLES = 200, FEATURES = 20;
        long timeStandard = 0, timeParallel = 0;
        
        Random random = new Random();
        AdaBoostStandardClassifier standard = new AdaBoostStandardClassifier(150);
        AdaBoostParallelClassifier parallel = new AdaBoostParallelClassifier(150);
        
        for (int num = 1; num <= 10; num++) {
            double[][] dataX = new double[SAMPLES][FEATURES];
            int[] dataY = new int[SAMPLES];
            for (int sample = 0; sample < SAMPLES; sample++) {
                dataY[sample] = random.nextBoolean()? -1: 1;
                for (int feature = 0; feature < FEATURES; feature++) {
                    dataX[sample][feature] = random.nextGaussian();
                }
            }
            long millis = CLOCK.millis();
            var resultStd = standard.fit(dataX, dataY);
            int[] predYStd = standard.predict(dataX);
            timeStandard += CLOCK.millis() - millis;
            millis = CLOCK.millis();
            var resultPrl = parallel.fit(dataX, dataY);
            int[] predYPrl = parallel.predict(dataX);
            timeParallel += CLOCK.millis() - millis;

            assertEquals(resultStd, resultPrl, "Different results");
            assertArrayEquals(predYStd, predYPrl, "Different predictions");
            log.info("Iteration {} completed", num);
        }
        log.info("Parallel version consumed {} msec; standard version consumed {} msec.", timeParallel, timeStandard);
    }
}
