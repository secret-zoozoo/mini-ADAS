#include <stdio.h>
#include <stdlib.h>   // atoi
#include <string.h>   // strcmp

int main(int argc, char *argv[])
{
    int a, b;
    int result;

    /* 인자 개수 체크 */
    if (argc != 4) {
        printf("Usage: %s <num1> <num2> <add|sub|mul|div>\n", argv[0]);
        return 1;
    }

    /* 문자열 → 정수 변환 */
    a = atoi(argv[1]);
    b = atoi(argv[2]);

    /* 연산자 처리 */
    if (strcmp(argv[3], "add") == 0) {
        result = a + b;
    } else if (strcmp(argv[3], "sub") == 0) {
        result = a - b;
    } else if (strcmp(argv[3], "mul") == 0) {
        result = a * b;
    } else if (strcmp(argv[3], "div") == 0) {
        if (b == 0) {
            printf("Error: division by zero\n");
            return 1;
        }
        result = a / b;
    } else {
        printf("Unknown operator: %s\n", argv[3]);
        return 1;
    }

    printf("Result: %d\n", result);
    return 0;
}