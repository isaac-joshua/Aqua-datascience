def merge_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output:
        # Use zip_longest to handle files of different lengths
        from itertools import zip_longest
        
        # Iterate over both files simultaneously
        for line1, line2 in zip_longest(file1, file2, fillvalue=''):
            # Strip newline characters and combine the lines
            merged_line = line1.strip() +' '+ line2.strip() + '\n'
            output.write(merged_line)

