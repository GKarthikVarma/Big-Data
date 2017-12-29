import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

class Vertex implements Writable{
	int tag;
	long group;
	long vid;
	String adjacent = "";
	
	Vertex(){}
	
	Vertex(int t, long g)
	{
		tag = t;
		group = g;
	}
	
	Vertex(int t, long g, long v, String a)
	{
		tag = t;
		group = g;
		vid = v;
		adjacent = a;
	}

	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
		tag = in.readInt();
		group = in.readLong();
		vid = in.readLong();
		adjacent = in.readLine();
		
	}

	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
		out.writeInt(tag);
		out.writeLong(group);
		out.writeLong(vid);
		out.writeBytes(adjacent);
	}
	
}

public class Graph extends Configured implements Tool {
	
	/*************************************
	 * First mapper and reducer job
	 ************************************/
	public static class Mapper_First extends Mapper<LongWritable,Text,LongWritable,Vertex>{
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			Scanner sc = new Scanner(value.toString()).useDelimiter(",");
			int vid = sc.nextInt();
			String adj =Integer.toString(sc.nextInt());
			while(sc.hasNext())
			{
				adj = adj + "," + sc.nextInt();
			}
			
			//make vertex to send
			Vertex v = new Vertex(0,vid,vid,adj);
			context.write(new LongWritable(v.vid), v);
		}
	}
	
	public static class Reducer_First extends Reducer<LongWritable,Vertex,Text,Text>
	{
		@Override
		public void reduce(LongWritable key, Iterable<Vertex> value,Context context) throws IOException,InterruptedException{
			for(Vertex v: value)
			{
				Text values = new Text();
				Text keys = new Text();
				keys.set(key.toString());
				values.set("," + v.tag + "," + v.group + "," + v.vid + "," + v.adjacent);
				context.write(keys, values);
			}
		}
	}
	
	/*****************************
	 * Iterative mapper and reducer jobs
	 ****************************/
	public static class Mapper_Iterative extends Mapper<LongWritable,Text,LongWritable,Vertex>{
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			Scanner sc = new Scanner(value.toString().replaceAll("\\s+", "")).useDelimiter(",");
			int keyz = sc.nextInt();
			int tag = sc.nextInt();
			int group = sc.nextInt();
			int vid = sc.nextInt();
			String adjacent = "";
			while(sc.hasNext())
			{
				if(adjacent.isEmpty())
				{
					adjacent = Integer.toString(sc.nextInt());
				}
				else
				{
					adjacent = adjacent + "," + Integer.toString(sc.nextInt());
				}
			}
			Vertex v = new Vertex(tag,group,vid,adjacent);
			context.write(new LongWritable(v.vid), v);
			
			if(!adjacent.isEmpty())
			{
				Scanner sc1 = new Scanner(adjacent).useDelimiter(",");
				while(sc1.hasNext())
				{
					Vertex v2 = new Vertex(1,v.group);
					context.write(new LongWritable(sc1.nextInt()), v2);
				}
			}
		}
	}
	
	public static class Reducer_Iterative extends Reducer<LongWritable,Vertex,Text,Text>
	{
		@Override
		public void reduce(LongWritable key, Iterable<Vertex> value,Context context) throws IOException,InterruptedException{
			long m = Long.MAX_VALUE;
			Text a = new Text();
			Text b = new Text();
			String adjacent = "";
			for(Vertex v:value)
			{
				if(v.tag == 0)
				{
					adjacent = v.adjacent;
				}
				m = Math.min(m, v.group);
			}
			a.set(Long.toString(m));
			Vertex v1 = new Vertex(0,m,key.get(),adjacent);
			b.set("," + 0 + "," + v1.group + "," + v1.vid + "," + v1.adjacent);
			context.write(a,b);
		}
	}
	
	/*******************************
	 * Final Mapper and Reducer
	 *******************************/
	public static class Mapper_Final extends Mapper<LongWritable,Text,LongWritable,IntWritable>{
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			Scanner sc = new Scanner(value.toString().replaceAll("\\s+", "")).useDelimiter(",");
			int group = sc.nextInt();
			context.write(new LongWritable(group), new IntWritable(1));
		}
	}
	
	public static class Reducer_Final extends Reducer<LongWritable,IntWritable,Text,Text>
	{
		@Override
		public void reduce(LongWritable key, Iterable<IntWritable> value,Context context) throws IOException,InterruptedException{
			int m = 0;
			for(IntWritable v:value)
			{
				m = m + v.get();
			}
			Text a = new Text();
			Text b = new Text();
			a.set(Long.toString(key.get()));
			b.set(Integer.toString(m));
			context.write(a,b);
		}
	}
	
	/*************************************
	 * Main Methods
	 ************************************/
    public static void main ( String[] args ) throws Exception {
    	int foo = ToolRunner.run(new Configuration(), new Graph(),args);
    	System.exit(foo);
    }
    public int run(String[] args)throws Exception
    {
    	/***********************************
    	 * Job 1
    	 **********************************/
    	Configuration conf = new Configuration();
    	Job job1 = Job.getInstance(conf, "Project 2");
    	job1.setJarByClass(Graph.class);
    	job1.setMapperClass(Mapper_First.class);
    	job1.setReducerClass(Reducer_First.class);
    	job1.setMapOutputKeyClass(LongWritable.class);
    	job1.setMapOutputValueClass(Vertex.class);
    	job1.setOutputKeyClass(Text.class);
    	job1.setOutputValueClass(Text.class);
    	FileInputFormat.addInputPath(job1, new Path(args[0]));
    	FileOutputFormat.setOutputPath(job1, new Path(args[1] + "/f0"));
    	job1.waitForCompletion(true);
		
		/************************************
		 * Job 2 Iterative
		 ***********************************/
		int iterations;
		for(iterations = 0; iterations < 5; iterations++)
		{
			Configuration conf_iterate = new Configuration();
	    	Job job_iterate = Job.getInstance(conf_iterate, "Project 2");
	    	job_iterate.setJarByClass(Graph.class);
	    	job_iterate.setMapperClass(Mapper_Iterative.class);
	    	job_iterate.setReducerClass(Reducer_Iterative.class);
	    	job_iterate.setMapOutputKeyClass(LongWritable.class);
	    	job_iterate.setMapOutputValueClass(Vertex.class);
	    	job_iterate.setOutputKeyClass(Text.class);
	    	job_iterate.setOutputValueClass(Text.class);
	    	FileInputFormat.addInputPath(job_iterate, new Path(args[1] + "/f" + iterations));
	    	FileOutputFormat.setOutputPath(job_iterate, new Path(args[1] + "/f" + (iterations + 1)));
	    	job_iterate.waitForCompletion(true);
		}
		
		/***********************************
		 * Final Job
		 **********************************/
		Configuration conf_final = new Configuration();
    	Job job_final = Job.getInstance(conf_final, "Project 2");
    	job_final.setJarByClass(Graph.class);
    	job_final.setMapperClass(Mapper_Final.class);
    	job_final.setReducerClass(Reducer_Final.class);
    	job_final.setMapOutputKeyClass(LongWritable.class);
    	job_final.setMapOutputValueClass(IntWritable.class);
    	job_final.setOutputKeyClass(Text.class);
    	job_final.setOutputValueClass(Text.class);
    	FileInputFormat.addInputPath(job_final, new Path(args[1] + "/f5"));
    	FileOutputFormat.setOutputPath(job_final, new Path(args[2]));
    	job_final.waitForCompletion(true);
    	
		return 0;
    }
}
