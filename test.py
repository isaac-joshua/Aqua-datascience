def merge_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output:
        # Use zip_longest to handle files of different lengths
        from itertools import zip_longest
        
        # Iterate over both files simultaneously
        for line1, line2 in zip_longest(file1, file2, fillvalue=''):
            # Strip newline characters and combine the lines
            merged_line = line1.strip() +' '+ line2.strip() + '\n'
            output.write(merged_line)

def removeverse(refernce_path, result_path):
    with open(refernce_path, 'r') as vref_file:
        vref_lines = vref_file.readlines()

    with open(result_path, 'r') as merged_file:
        merged_lines = merged_file.readlines()

    # Extract references to look for
    vref_references = [line.strip() for line in vref_lines]

    # Open the merged results file to write the updated content
    with open('data/merged_results.txt', 'w') as merged_file_updated:
        for line in merged_lines:
            if any(ref in line for ref in vref_references):
                merged_file_updated.write('\n')  # Write a blank line
            else:
                merged_file_updated.write(line)  # Write the original line

    print("The references have been replaced with blank lines in the updated file.")
