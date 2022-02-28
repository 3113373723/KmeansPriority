package PriorityIteration;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;

// 第一遍fullBatch,然后lazybatch和minibatch交替
public class Kmeans2 {

    public static void main(String args[]) throws Exception {
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
        Date date = new Date(System.currentTimeMillis());

//		String input = "/home/tuyilei/module/dataset/initialData/HIGGS.csv";
//		String output = "/home/zhongliangsheng/module/code/output/secIte-"+formatter.format(date);
        String input = "E:\\graduationProject\\KMeans2\\src\\data-2022-02-13-22-24-31.cvs";
        String output = "E:\\graduationProject\\KMeans2\\out\\lazyLog-" + formatter.format(date) + ".txt";

        File file = new File(output);
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw = new BufferedWriter(fw);

        int numOfCenters = 100; //质心数
        int numOfIterations = 1000; //总迭代数
        int batchSize = 10000; //每个batch的数量

        ArrayList<ArrayList<Double>> points = load(input); //加载数据

        int[] tags = new int[points.size()]; //记录每个point的质心
        Arrays.fill(tags, -1);

        ArrayList<Double>[] centers = initCenters(points, numOfCenters); //初始化质心，取前numOfCenters个

        int counterOfIterations = 1; //迭代数
        int hasPro = 0; //一个batch内已经处理过多少数据
        int numOfDimensions = points.get(0).size();
        double[][] aggOfSum = new double[numOfCenters][numOfDimensions]; //每个质心的数据和
        int[] aggOfCounter = new int[numOfCenters]; //每个质心下的point的数量
        double batchError = 0.0f, baseErr = 0.0f, currBaseErr = 0.0f;
        double dist, minDist;
        int tag, changeTagCount = 0;
        int dataProCounters = 0; //数据被处理了多少遍
        int CntOfLazy; //Lazy EM每次执行的次数
        ArrayList<Double>[] LazyCenters = centers; //定义lazyEM质心
        double[] laseErr = new double[points.size()];
        boolean isContinue = true; //设置一个变量控制迭代是否继续，当达到收敛值时改为false

        /** 获取当前系统时间*/
        long startTime = System.currentTimeMillis();

        //开始迭代
        while (counterOfIterations < numOfIterations && isContinue) {

            dataProCounters++;
            // FULL EM
            for (int index = 0; index < points.size(); index++) { //遍历全部数据
                if (counterOfIterations > numOfIterations) break;
                //if(toHalt[index]) continue;
                // Expectation step
                minDist = Double.MAX_VALUE;
                tag = -1;
                for (int i = 0; i < numOfCenters; i++) { // for a point, find minDistance and change;
                    dist = distance(points.get(index), centers[i]);
                    if (dist < minDist) {
                        minDist = dist;
                        tag = i;
                    }
                }//for distance

                if (tags[index] != tag) {
                    aggregate(aggOfSum, aggOfCounter, points.get(index),
                            tag, tags[index], dataProCounters);
                    tags[index] = tag;
                    changeTagCount++;
                }
                batchError += minDist;
                hasPro++;
                if (counterOfIterations > 1) {
                    baseErr += (minDist - laseErr[index]);
                }
                laseErr[index] = minDist;

                if (hasPro == batchSize && dataProCounters > 1) { //处理数量达到一个batch的数量，进行一次更新
                    System.out.println(" Iteration is : " + counterOfIterations + " point change tag counts is : " + changeTagCount + " this batch errSum is : " + batchError + " errSum is : " + baseErr);
                    bw.write(" Iteration is : " + counterOfIterations + "point change tag counts is : " + changeTagCount + " this batch errSum is : " + batchError + " errSum is : " + baseErr);
                    bw.newLine();
                    batchError = 0;
                    hasPro = 0;
                    changeTagCount = 0;
                    // 应该没问题，第一遍full EM不会进入到这，以后的才会，每次处理达到一个batch的数量counterOfIterations就会加一
                }
            }//for
            if (baseErr < 315658) isContinue = false;
            // 不明白为什么updateCenters(M-step)只在dataProCounters==1时运行过一次，个人认为每遍都需要更新
            if (dataProCounters == 1) {
                System.out.println(" Iteration is : " + counterOfIterations + " point change tag counts is : " + changeTagCount + " this batch errSum is : " + batchError + " errSum is : " + baseErr);
                bw.write(" Iteration is : " + counterOfIterations + "point change tag counts is : " + changeTagCount + " this batch errSum is : " + batchError + " errSum is : " + baseErr);
                bw.newLine();
                baseErr = batchError;
                batchError = 0;
                hasPro = 0;
                changeTagCount = 0;
            }
            counterOfIterations++;
            centers = updateCenters(aggOfSum, aggOfCounter);//Maximization step
            // Lazy EM
            // 正常来说应该是先E再M，但是第一遍E没有意义（Dn-1==Dn),so 先M，之后再EM这样进行
            // E就是计算距离，M就是确定质心
            // 每次Full EM后要进行若干次lazy EM，可设置次数
            CntOfLazy = 2;

            while (CntOfLazy != 0 && counterOfIterations < numOfIterations && isContinue) {
                CntOfLazy--;
                dataProCounters++;
                double AverageDis = 0;
                // 计算平均最小距离，用来区分数据优先级
                for (double Dis : laseErr) {
                    AverageDis += Dis;
                }
                AverageDis /= laseErr.length;//这里用的是小于averageDIs的数据为优先级高的，也可以乘以某个参数进一步限制数据点个数
                for (int index = 0; index < points.size(); index++) { //遍历全部数据
                    if (laseErr[index] > AverageDis) continue;
                    minDist = Double.MAX_VALUE;
                    tag = -1;
                    for (int i = 0; i < numOfCenters; i++) { // for a point, find minDistance and change;
                        dist = distance(points.get(index), centers[i]);
                        if (dist < minDist) {
                            minDist = dist;
                            tag = i;
                        }
                    }//for distance

                    if (tags[index] != tag) {//如果tag发生变化
                        aggregate(aggOfSum, aggOfCounter, points.get(index),
                                tag, tags[index], dataProCounters);
                        // 其实LazyAggOfSum和LazyAggOfCounter也应该变
                        // 但是本轮lazyEM用不到了故省略,tag变了之后，新一轮lazy EM重新定义它们时会相应改变
                        tags[index] = tag;
                        changeTagCount++;
                    }
                    batchError += minDist;
                    hasPro++;
                    if (counterOfIterations > 1) {
                        // 更新
                        baseErr += (minDist - laseErr[index]);
                    }
                    // 更新最短距离
                    laseErr[index] = minDist;
                }// end E-step
                updateCenters(aggOfSum, aggOfCounter);
                System.out.println(" Iteration is : " + counterOfIterations + " point change tag counts is : " + changeTagCount + " this batch errSum is : " + batchError + " errSum is : " + baseErr);
                bw.write(" Iteration is : " + counterOfIterations + "point change tag counts is : " + changeTagCount + " this batch errSum is : " + batchError + " errSum is : " + baseErr);
                bw.newLine();
                batchError = 0;
                hasPro = 0;
                changeTagCount = 0;
                counterOfIterations++;
                if (baseErr < 315658) isContinue = false;
            }// end Lazy EM

        } //while
        bw.write(stringOfCenters(counterOfIterations, centers));
        /** 获取当前的系统时间，与初始时间相减就是程序运行的毫秒数，除以1000就是秒数*/
        long endTime = System.currentTimeMillis();
        double usedTime = (endTime - startTime) / (double) 1000;
        System.out.println("Total Time is : " + usedTime + "s");
        bw.write(" Total Time is : " + usedTime + "s");
        bw.close();
        fw.close();
    }

    /**
     * Load data points.
     *
     * @param input
     * @throws Exception
     */
    private static ArrayList<ArrayList<Double>> load(String input)
            throws Exception {
        ArrayList<ArrayList<Double>> points = new ArrayList<ArrayList<Double>>();
        File inputF = new File(input);
        FileReader fr = new FileReader(inputF);
        BufferedReader br = new BufferedReader(fr);

        String context = null;
        int count = 0;
        while ((context = br.readLine()) != null) {
            String[] dims = context.split(",");
            ArrayList<Double> point = new ArrayList<Double>();
			/*for (String dim: dims) {
				point.add(Double.parseDouble(dim));
			}*/
            for (int i = 1; i < dims.length; i++) {
                point.add(Double.parseDouble(dims[i]));
            }
            points.add(point);
            count++;
            if (count % 1000000 == 0) {

                System.out.println("has load count : " + (count));
            }
        }
        System.out.println("load " + points.size() + " points successfully!");

        return points;
    }

    /**
     * Initialize centers before computation.
     *
     * @param points
     * @param numOfCenters
     * @return
     */
    private static ArrayList<Double>[] initCenters(   //initialize center as  first k points
                                                      ArrayList<ArrayList<Double>> points, int numOfCenters) {
        ArrayList<Double>[] centers = new ArrayList[numOfCenters];
        for (int i = 0; i < numOfCenters; i++) {
            ArrayList<Double> center = new ArrayList<Double>();
            for (double dim : points.get(i)) {
                center.add(dim);
            }
            centers[i] = center;
        }
        return centers;
    }

    /**
     * Compute the distance of two points.
     *
     * @param point
     * @param center
     * @return
     */
    private static double distance(ArrayList<Double> point, ArrayList<Double> center) {
        double sum = 0.0f;
        int numOfDim = point.size();
        for (int i = 0; i < numOfDim; i++) {
            sum = sum + (point.get(i) - center.get(i)) * (point.get(i) - center.get(i));
        }
        return Math.sqrt(sum);
    }

    private static void aggregate(double[][] aggOfSum, int[] aggOfCounter,
                                  ArrayList<Double> point, int tag, int oldTag, int dataProCounters) {
        int i = 0;
        for (Double dim : point) {//for this point's every dimension
            aggOfSum[tag][i] += dim;
            if (dataProCounters > 1) {//judge is the first execute or not
                aggOfSum[oldTag][i] -= dim;
            }
            i++;
        }
        aggOfCounter[tag]++;
        if (dataProCounters > 1) {
            aggOfCounter[oldTag]--;
        }
    }

    private static void LazyAggregate(double[][] LazyAggOfSum, int[] LazyAggOfCounter,
                                      ArrayList<Double> point, int tag) {
        int i = 0;
        for (Double dim : point) {//for this point's every dimension
            LazyAggOfSum[tag][i] += dim;
            i++;
        }
        LazyAggOfCounter[tag]++;
    }


    private static ArrayList<Double>[] updateCenters(double[][] aggOfSum,
                                                     int[] aggOfCounter) {
        int numOfCenters = aggOfSum.length;
        int numOfDim = aggOfSum[0].length;
        ArrayList<Double>[] centers = new ArrayList[numOfCenters];
        for (int i = 0; i < numOfCenters; i++) {
            ArrayList<Double> center = new ArrayList<Double>();
            for (int j = 0; j < numOfDim; j++) {
                center.add(aggOfSum[i][j] / aggOfCounter[i]);
            }
            centers[i] = center;
        }
        return centers;
    }

    private static ArrayList<Double>[] LazyUpdateCenters(double[][] LazyAggOfSum,
                                                         int[] LazyAggOfCounter) {
        int numOfCenters = LazyAggOfSum.length;
        int numOfDim = LazyAggOfSum[0].length;
        ArrayList<Double>[] centers = new ArrayList[numOfCenters];
        for (int i = 0; i < numOfCenters; i++) {
            if (LazyAggOfCounter[i] > 0) {
                ArrayList<Double> center = new ArrayList<Double>();
                for (int j = 0; j < numOfDim; j++) {
                    center.add(LazyAggOfSum[i][j] / LazyAggOfCounter[i]);
                }
                centers[i] = center;
            }
        }
        return centers;
    }

    private static String stringOfCenters(int counterOfIterations,
                                          ArrayList<Double>[] centers) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < centers.length; i++) {
            sb.append("{");
            sb.append(Integer.toString(i));
            sb.append(", [");
            for (double dim : centers[i]) {
                sb.append(Double.toString(dim));
                sb.append(",");
            }
            sb.append("]}");
            sb.append("\n");
        }
        return sb.toString();
    }
}