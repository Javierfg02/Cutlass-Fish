# Notes

## Preprocessing data

### Their data

They have 3 files that they join together:

- src: Each line of this file is a sentence
- trg: Each line of this file is a sequence of frames. That is, each line is one clip (I think) where each frame always has 150 data . points. They also append a counter at the end of each frame which delimits the position of that frame relative to the whole sentence
- files: Each line of this file is a file name linking to some data. I don't particularly understand this file

### Our data (as of April 17th)

We have the following 2 functions that return dictionaries:

- create_trg: Maps {Sentence ID: [[frame data for json file 1],
                                    [frame data for json file 2],
                                    ... ,
                                    [frame data for last json file in clip]]}

                That is, it maps a sentence ID to a 2D array where each element contains the data for 1 frame
                Each of ours frames has 

- create_src: Maps {Sentence ID: setnence as a string}. Pretty intuitive.
