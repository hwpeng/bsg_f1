#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <stdlib.h>

// our software stack
#include "../host.h"
#include "../device.h"
#include "driver/bsg_dma_driver.h"

#define debug 0

static const uint32_t BUF_SIZE = 4 * 1024 * 1024;
static const uint32_t ALIGNMENT = 64;
static const uint32_t POP_SIZE = 64;

static char *dev_path = "/dev/bsg_dma_driver";
int dev_fd; 

static uint32_t get_tail (struct Host *host) {
	uint32_t tail = 0;
	if (ioctl(dev_fd, IOCTL_TAIL, &tail) != 0 ) // error
		printf("ioctl error. returning tail as 0.\n");
	return tail; 
}

static void write_wr_addr_high (struct Host *host, uint32_t val) {	
	if (ioctl(dev_fd, IOCTL_WR_ADDR_HIGH, val) != 0) 
		printf("ioctl error.\n");
}

static void write_wr_addr_low (struct Host *host, uint32_t val) {
	if (ioctl(dev_fd, IOCTL_WR_ADDR_LOW, val) != 0) 
		printf("ioctl error.\n");
}

static void write_wr_head (struct Host *host, uint32_t val) {
	if (ioctl(dev_fd, IOCTL_WR_HEAD, val) != 0) 
		printf("ioctl error.\n"); 
}

static void write_wr_len (struct Host *host, uint32_t val) {
	if (ioctl(dev_fd, IOCTL_WR_LEN, val) != 0) 
		printf("ioctl error.\n"); 
}

static void write_wr_buf_size (struct Host *host, uint32_t val) {
	if (ioctl(dev_fd, IOCTL_WR_BUF_SIZE, val) != 0) 
		printf("ioctl error.\n"); 
}

static void start_write (struct Host *host) {
	if (ioctl(dev_fd, IOCTL_CFG, 0x10) != 0) 
		printf("ioctl error when writing to CFG_REG.\n"); 
	if (ioctl(dev_fd, IOCTL_CNTL, 0x1) != 0) 
		printf("ioctl error when writing to CNTL_REG.\n"); 
}

static void stop (struct Host *host) {
	if (ioctl(dev_fd, IOCTL_CNTL, STOP) != 0) 
		printf("ioctl error.\n"); 
}

/*
 * TODO: return read's return value. 
 * */
static bool pop (struct Host *host, uint32_t pop_size) {
	if (read(dev_fd, host->buf_cpy, pop_size))		
		return false;
	return true;
} 

/* 
 * prints data as a sequence of unsigned chars.
 * */
void print(struct Host *host, uint32_t ofs, uint32_t size) {
	if (!host->buf_cpy)
		printf("Host::print: buf_cpy is null. can't print.\n");
	else if (!size)
		printf("Host::print: size is 0. can't print.\n");
	else if (ofs + size > host->buf_size)
		printf("Host::print: invalid range to print. can't print. \n");

	for (int i = ofs; i < size; i++) {
		printf("0x%02X", host->buf_cpy[i]);
		if ((i + 1) % 10 == 0)
			printf("\n");
		else
			printf(" ");
	}
	printf("\n");
}

void deploy_init_host (struct Host *host, uint32_t buf_size, uint32_t align) {
	dev_fd = open(dev_path, O_RDONLY);
	if (dev_fd == -1) {
		printf("Unable to open device.\n");
		return; 
	}
	host->buf_size = DMA_BUFFER_SIZE; /* global buffer */
	host->buf = NULL; /* TODO: rename to mmap_buf */
	ioctl(dev_fd, IOCTL_CLEAR_BUFFER);
	host->buf_cpy = (char *) aligned_alloc(align, buf_size + 64); /* user copy of buffer */
	memset(host->buf_cpy, 0, buf_size + 64);
	
	host->head = 0;
	
	host->get_tail = get_tail;	
	host->write_wr_addr_high = write_wr_addr_high;
	host->write_wr_addr_low = write_wr_addr_low;
	host->write_wr_head = write_wr_head;
	host->write_wr_len = write_wr_len;
	host->write_wr_buf_size = write_wr_buf_size;
	host->start_write = start_write;
	host->stop = stop;
	host->pop = pop;
	host->print = print;
}
