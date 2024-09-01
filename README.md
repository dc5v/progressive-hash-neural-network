# Vision Tiles Hashing Neural Network (VTHNN) 

- 영상 타일 분할 Generate Tiles

- 해시 값 생성 Tiles Hashing 

- 타일 해시 배열 생성 Hash Array Formatter


`코드개료`

```c
#include <stdio.h>
#include <opencv2/opencv.hpp>

unsigned long gen_hash ( const unsigned char *data, int length, unsigned long prime )
{
  unsigned long hash = 0;

  for ( int i = 0; i < length; i++ )
  {
    hash = hash * prime + data[i];
  }
  return hash;
}

void process_frame ( cv::Mat frame, int rows, int cols, unsigned long prime )
{
  int h = frame.rows / rows;
  int w = frame.cols / cols;

  for ( int i = 0; i < rows; i++ )
  {
    for ( int j = 0; j < cols; j++ )
    {
      cv::Rect rect ( j * w, i * h, w, h );
      cv::Mat  tile = frame ( rect );

      cv::Mat gtile;
      cv::cvtColor ( tile, gtile, cv::COLOR_BGR2GRAY );

      int            len  = gtile.total () * gtile.elemSize ();
      unsigned char *data = gtile.data;
      unsigned long  hash = gen_hash ( data, len, prime );

      printf ( "tile %d-%d, hash %lu \n", i, j, hash );
    }
  }
}

void process_video ( const char *filename, int rows, int cols, unsigned long prime )
{
  cv::VideoCapture c ( filename );

  if ( !c.isOpened () )
  {
    return;
  }

  while ( 1 )
  {
    cv::Mat frame;
    c >> frame;

    if ( frame.empty () )
    {
      break;
    }

    process_frame ( frame, rows, cols, prime );
  }

  cap.release ();
}

int main ( int argc, char **argv )
{
  if ( argc < 2 )
  {
    printf ( "Usage: %s [filename] \n", argv[0] );
    return -1;
  }

  int           trows = 10;
  int           tcols = 10;
  unsigned long prime = 31; // PRIME NUMBER

  process_video ( argv[1], trows, tcols, prime );

  return 0;
}

```
