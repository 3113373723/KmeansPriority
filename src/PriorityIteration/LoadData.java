package PriorityIteration;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class LoadData {
    public static void main(String[] args) throws Exception{
        SimpleDateFormat formatter= new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
        Date date = new Date(System.currentTimeMillis());

        String input = "/home/tuyilei/module/dataset/initialData/HIGGS.csv";
        String output = "/home/zhongliangsheng/module/code/output/data-"+formatter.format(date)+".cvs";


        File inputF = new File(input);
        FileReader fr = null;
        BufferedReader br = null;

        File file = new File(output);
        FileWriter fw = null;
        BufferedWriter bw = null;

        try {
            fr = new FileReader(inputF);
            br = new BufferedReader(fr);
            fw = new FileWriter(file);
            bw = new BufferedWriter(fw);
            String context = null;
//        String line = System.getProperty("line.separator"); //在这个位置更换为自己想使用的换行符
            int count = 0;
            while((context=br.readLine()) != null) {
                bw.write(context);
                bw.newLine();
                count++;
                if(count == 1000000) {
                    break;
                }
            }// end while
        }catch (FileNotFoundException e ) {
            e.printStackTrace();
        }catch (IOException e){
            e.printStackTrace();
        }finally {
            br.close();
            fr.close();
            bw.close();
            fw.close();
        }



    }
}
