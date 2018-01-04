import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

class Matrix_M implements Writable{
	 public int M_i,M_j;
	 public float M_value;
	 
	 Matrix_M() {}
	 
	 Matrix_M(int x, int y, float z)
	 {
		 M_i = x;
		 M_j = y;
		 M_value = z;
	 }

	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		M_i = in.readInt();
		M_j = in.readInt();
		M_value = in.readFloat();
	}

	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(M_i);
		out.writeInt(M_j);
		out.writeFloat(M_value);
	}
	
}

class Matrix_N implements Writable{
	
	public int N_i,N_j;
	public float N_value;
	
	Matrix_N() {}
	 
	 Matrix_N(int x, int y, float z)
	 {
		 N_i = x;
		 N_j = y;
		 N_value = z;
	 }

	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		N_i = in.readInt();
		N_j = in.readInt();
		N_value = in.readFloat();
	}

	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(N_i);
		out.writeInt(N_j);
		out.writeFloat(N_value);
	}
	
}

class Matrix_MN implements Writable{
	public int i,j;
	public float value;
	
	Matrix_MN() {}
	Matrix_MN(int x, int y, float z)
	{
		i = x;
		j = y;
		value = z;
	}
	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		i = in.readInt();
		j = in.readInt();
		value = in.readFloat();
	}
	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(i);
		out.writeInt(j);
		out.writeFloat(value);
	}
}

class Matrix_intermediate implements Writable{

	public char tag;
	public Matrix_M m1;
	public Matrix_N n1;
	
	Matrix_intermediate() {}
	Matrix_intermediate(Matrix_M m2)
	{
		tag = 'M'; 
		m1 = m2;
	}
	
	Matrix_intermediate(Matrix_N n2)
	{
		tag = 'N';
		n1 = n2;
	}
	
	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		tag = in.readChar();
		if(tag == 'M')
		{
			m1 = new Matrix_M();
			m1.readFields(in);
		}else {
			n1 = new Matrix_N();
			n1.readFields(in);
		}
		
	}

	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeChar(tag);
		if(tag == 'M')
		{
			m1.write(out);
		}else {
			n1.write(out);
		}
	}
	
}

class Matrix_Final implements Writable{

	public Matrix_MN mn1;
	public int i,j;
	public float value;
	
	Matrix_Final() {}
	Matrix_Final(Matrix_MN mn2)
	{
		mn1 = mn2;
	}
	
	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		i = in.readInt();
		j = in.readInt();
		value = in.readFloat();
	}
	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(i);
		out.writeInt(j);
		out.writeFloat(value);
	}

}



public class Multiply extends Configured implements Tool {
	
	
	//Matrix M Mapper
	public static class Mapper1 extends Mapper<LongWritable,Text,Text,Matrix_intermediate>{
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			//Get next line 
			String next_line = value.toString();
			String[] val = next_line.split(",");
			Matrix_M m1 = new Matrix_M(Integer.parseInt(val[0]),Integer.parseInt(val[1]),Float.parseFloat(val[2]));
			Text key_pair = new Text();
			key_pair.set(String.valueOf(m1.M_j));
			context.write(key_pair, new Matrix_intermediate(m1));
			
		}
		
	}
	
	//Matrix N Mapper
	public static class Mapper2 extends Mapper<LongWritable,Text,Text,Matrix_intermediate>{
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			String next_line = value.toString();
			String[] val = next_line.split(",");
			Matrix_N n1 = new Matrix_N(Integer.parseInt(val[0]),Integer.parseInt(val[1]),Float.parseFloat(val[2]));
			Text key_pair = new Text();
			key_pair.set(String.valueOf(n1.N_i));
			context.write(key_pair, new Matrix_intermediate(n1));
		}
		
	}
	
	public static class Mapper3 extends Mapper<LongWritable,Text,Text,Text>{
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			String next_line = value.toString();
			String[] val = next_line.split(",");
			Text key_pair = new Text();
			key_pair.set(val[1] + "," + val[2]);
			Text value_pair = new Text();
			value_pair.set(val[3]);			
			context.write(key_pair, value_pair);
		}
	}
	
	
	public static class Reducer_multiply extends Reducer<Text,Matrix_intermediate,Text,Text>{
		@Override
		public void reduce(Text key, Iterable<Matrix_intermediate> value, Context context)throws IOException, InterruptedException{
			String input_reducer = value.toString();
			String[] input_values;
			
			//getting inputs to reducer and dividing into two parts
			Vector<Matrix_M> M_matrix = new Vector<Matrix_M>();
			Vector<Matrix_N> N_matrix = new Vector<Matrix_N>();
			Text key_pair = new Text();
			Text value_pair = new Text();
			double value_intermediate;
			
			M_matrix.clear();
			N_matrix.clear();
			
			for (Matrix_intermediate v : value)
			{
				if(v.tag == 'M')
				{
					M_matrix.add(v.m1);
				}else {
					N_matrix.add(v.n1);
				}
			}
			
			//Multiply values
			for(Matrix_M v: M_matrix)
			{
				for(Matrix_N v1:N_matrix)
				{
					key_pair.set(key + ",");
					value_intermediate = v.M_value * v1.N_value;
					value_pair.set(v.M_i + "," + v1.N_j + "," + value_intermediate);
					context.write(key_pair, value_pair);
				}
			}
		}
	}
	
	public static class Reducer_combine extends Reducer<Text,Text,Text,Text>{
		@Override
		public void reduce(Text key, Iterable<Text> value, Context context)throws IOException, InterruptedException{
			float output = 0;
		
			//Set Outputs
			Text key_pair = new Text();
			Text value_pair = new Text();
			
			for(Text v: value)
			{
				output += Float.parseFloat(v.toString());
			}
			key_pair.set(key + ",");
			value_pair.set(String.valueOf(output));
			context.write(key_pair, value_pair);
			
		}
	}

    public static void main ( String[] args ) throws Exception {
    	
    	int res = ToolRunner.run(new Configuration(), new Multiply(),args);
    	System.exit(res);
    }
    public int run(String[] args)throws Exception{
    	
    	Configuration conf = new Configuration();
    	
    	//Configure job 1
    	Job job1 = Job.getInstance(conf,"Matrix Multiplication Using Mapreduce");
    	job1.setJarByClass(Multiply.class);
    	job1.setOutputKeyClass(Text.class);
    	job1.setOutputValueClass(Matrix_intermediate.class);
    	
    	//set mapper and reducer
    	//Multiple inputs
    	MultipleInputs.addInputPath(job1,new Path(args[0]),TextInputFormat.class,Mapper1.class);
    	MultipleInputs.addInputPath(job1,new Path(args[1]),TextInputFormat.class,Mapper2.class);
    	
    	//Reducer config
    	FileOutputFormat.setOutputPath(job1,new Path(args[2], "out1"));
    	job1.setReducerClass(Reducer_multiply.class);

    	//End Job
    	job1.waitForCompletion(true);
    	
    	//Job 2 To build matrix
    	
    	Configuration conf2 = new Configuration();

    	Job job2 = Job.getInstance(conf2,"Forming the end matrix");
    	job2.setJarByClass(Multiply.class);
    	job2.setOutputKeyClass(Text.class);
    	job2.setOutputValueClass(Text.class);
    	
    	//set mapper and reducer
    	job2.setMapperClass(Mapper3.class);
    	job2.setReducerClass(Reducer_combine.class);
    	
    	//set input and output paths
    	FileInputFormat.addInputPath(job2, new Path(args[2],"out1"));
    	FileOutputFormat.setOutputPath(job2, new Path(args[3]));
    	return job2.waitForCompletion(true) ? 0 : 1;
    	
    	
    }
}
