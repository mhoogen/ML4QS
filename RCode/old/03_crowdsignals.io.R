# Load datasets

base_dir = "/Users/markhoogendoorn/Dropbox/Quantified-Self-Book/datasets/crowdsignals.io/Csv-merged/";
accelerometer_phone_data = read.csv(paste(base_dir, "merged_accelerometer_1466120659000000000_1466125720000000000.csv", sep=""));
app_usage_data = read.csv(paste(base_dir, "merged_apps_1466120659731000000_1466125749000000000.csv", sep=""));
labels_data = read.csv(paste(base_dir, "merged_interval_label_1466120784502000000_1466125692364000000.csv", sep=""));
full_data = data.frame(time = integer(0), accelerometer_phone_x = numeric(0), accelerometer_phone_y = numeric(0), accelerometer_phone_z = numeric(0));

# Set some values wrt the "default" table we are going to build. We are going to aggregate some values there.

aggregation_window = 1000000000 # go to seconds
s_time = min(accelerometer_phone_data$timestamps);
e_time = max(accelerometer_phone_data$timestamps);
user = 0;
row_limit = 1000;

current_time = s_time
for (i in 1:min(floor((e_time - s_time)/aggregation_window), row_limit)){
  full_data[i,] = c(time = current_time, accelerometer_phone_x = NA, accelerometer_phone_y = NA, accelerometer_phone_z = NA);
  current_time = current_time + aggregation_window;
}

# Add all accelerometer data

for (i in 1:nrow(full_data)){
  start_time = full_data$time[i]
  end_time = start_time + aggregation_window
  
  # Get all accelerometer data measured during the time frame
  acc_data = subset(accelerometer_phone_data, timestamps >= start_time & timestamps < end_time);
  
  # And average the values.
  
  full_data$accelerometer_phone_x[i] = mean(acc_data$x);
  full_data$accelerometer_phone_y[i] = mean(acc_data$y);
  full_data$accelerometer_phone_z[i] = mean(acc_data$z);
}

# Create a column (i.e. attribute) for each app we measure something about. Set the default usage time to 0.

full_data_with_app_usage <- categorial_to_columns(full_data, app_usage_data, 'app', 'start', 'end');
full_data_with_intervals <- categorial_to_columns(full_data_with_app_usage, labels_data, 'label', 'start', 'end');

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

