# Load datasets

#setwd("../datasets/crowdsignals.io/Csv-merged/")
rm(list=ls())
accelerometer = read.csv("merged_accelerometer_1466120659000000000_1466125720000000000.csv")
app_usage_data = read.csv("merged_apps_1466120659731000000_1466125749000000000.csv")
labels_data = read.csv("merged_interval_label_1466120784502000000_1466125692364000000.csv")

# Set some values wrt the "default" table we are going to build. We are going to aggregate some values there.
tics2sec = 1000000000
timeAtts = c("start","end","timestamps")
accelerometer[,timeAtts] = accelerometer[,timeAtts] / tics2sec
app_usage_data[,timeAtts] = app_usage_data[,timeAtts] / tics2sec

aggregation_window = 1 # go to seconds
s_time = min(accelerometer$timestamps)
e_time = max(accelerometer$timestamps)
current_time = s_time

user = 0
row_limit = 1000
N = min(floor((e_time - s_time)/aggregation_window), row_limit)
accelerometer$aggWindow = floor((accelerometer$timestamps-s_time)/aggregation_window)
app_usage_data$aggWindow = floor((app_usage_data$timestamps-s_time)/aggregation_window)

temp = accelerometer[accelerometer$aggWindow<N,c("x","y","z","aggWindow")]
temp_agg = aggregate(temp,by=list(temp$aggWindow),FUN=mean,na.rm=TRUE)
full_data = data.frame(time = current_time + (0:(N-1))* aggregation_window, 
                       acc_phone_x = temp_agg$x, acc_phone_y = temp_agg$y, 
                       acc_phone_z = temp_agg$z)

# Create columns of rows that describe a certain usage time (with start and end time) of a certain category.
categorial_to_columns = function(data_table, table_with_categories, cat_attribute, start_t_attribute, end_t_attribute){
  new_data_table = data_table;
  col = match(cat_attribute, names(table_with_categories));
  values_available = unique(table_with_categories[,col]);
  
  # Create a column for each value, remove undesired characters
  for (i in 1:length(values_available)){
    cleaned_name <- gsub("[^[:alnum:]]", "", values_available[i]);
    print("adding ....");
    print(cleaned_name);
    flush.console()
    new_data_table[paste(cat_attribute,cleaned_name, sep="_")] <- 0;
  }
  
  start_time_attribute = match(start_t_attribute, names(table_with_categories));
  end_time_attribute =match(end_t_attribute, names(table_with_categories));
  
  # And fill the rows with the appropriate values
  for (r in 1:nrow(table_with_categories)){
    start_time = table_with_categories[r, start_time_attribute];
    end_time = table_with_categories[r, end_time_attribute];
    
    # Get the matching records wrt time from our full dataset
    
    acc_data = subset(new_data_table, time >= (start_time - aggregation_window) & time <= end_time)
    
    if (nrow(acc_data) > 0){
      cleaned = paste(cat_attribute, gsub("[^[:alnum:]]", "", table_with_categories[r,col]), sep="_");
      new_col = match(cleaned, names(new_data_table));
      for (row in acc_data$time){
        new_row = match(row, new_data_table$time);
        new_data_table[new_row, new_col] = new_data_table[new_row, new_col] + 1;
      }
    }
  }
  return(new_data_table);
}

# Create a column (i.e. attribute) for each app we measure something about. Set the default usage time to 0.

full_data_with_app_usage <- categorial_to_columns(full_data, app_usage_data, 'app', 'start', 'end')
full_data_with_intervals <- categorial_to_columns(full_data_with_app_usage, labels_data, 'label', 'start', 'end')
