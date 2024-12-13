#!/bin/bash

# Directory containing PDF files
directory="./files/"

# Print headers
echo -e "File\tTitle\tAuthor\tSubject\tKeywords\tCreator\tProducer\tCreationDate\tModDate\tTagged\tPages\tEncrypted\tPageSize\tFileSize\tOptimized\tPDFVersion"

# Find PDF files and extract metadata, ignoring errors
find "$directory" -name "*.pdf" | while read -r file; do
  # Extract metadata using pdfinfo, ignoring errors
  pdfinfo "$file" 2>/dev/null | awk -v file="$file" '
    BEGIN { FS = ": "; OFS = "\t"; title=""; author=""; subject=""; keywords=""; creator=""; producer=""; creationdate=""; moddate=""; tagged=""; pages=""; encrypted=""; pagesize=""; filesize=""; optimized=""; pdfversion="" }
    /Title:/ { title=$2 }
    /Author:/ { author=$2 }
    /Subject:/ { subject=$2 }
    /Keywords:/ { keywords=$2 }
    /Creator:/ { creator=$2 }
    /Producer:/ { producer=$2 }
    /CreationDate:/ { creationdate=$2 }
    /ModDate:/ { moddate=$2 }
    /Tagged:/ { tagged=$2 }
    /Pages:/ { pages=$2 }
    /Encrypted:/ { encrypted=$2 }
    /Page size:/ { pagesize=$2 }
    /File size:/ { filesize=$2 }
    /Optimized:/ { optimized=$2 }
    /PDF version:/ { pdfversion=$2 }
    END {
      print file, title, author, subject, keywords, creator, producer, creationdate, moddate, tagged, pages, encrypted, pagesize, filesize, optimized, pdfversion
    }'
done

