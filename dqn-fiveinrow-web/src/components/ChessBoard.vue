<template>
  <div class="board" @click="handleClick">
    <div class="row" v-for="(row, rowIndex) in board" :key="rowIndex">
      <div class="cell" v-for="(cell, colIndex) in row" :key="colIndex"
           :style="{ width: cellSize + 'px', height: cellSize + 'px' }"
           @click="handleClickOnCell(rowIndex, colIndex)">
        <div class="chess" :class="cell ? (cell === 1 ? 'black' : 'white') : 'empty'"
             :style="{ width: cellSize + 'px', height: cellSize + 'px' }" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';


const props = withDefaults(
  defineProps<{
    cellClick: (row: number, col: number) => unknown;
    // boardSize: number,
    cellSize?: number,
    board: number[][],
  }>(),
  {
    boardSize: 11,
    cellSize: 40,
  }
);

const cellSize = props.cellSize;
// const boardSize = props.boardSize;
const board = props.board;

const handleClickOnCell = (row: number, col: number) => {
  // putChess(row, col)
  props.cellClick(row, col)
}

const handleClick = (e: MouseEvent) => {


};

const putChess = (row: number, col: number) => {
  // 如果该位置没有棋子，则落子
  if (board.value[row][col] === 0) {
    board.value[row][col] = 1; // 人类玩家落子
    // 这里你可以调用AI做出回应
  }
}
</script>

<style scoped>
.board {
  width: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  /* border: 2px solid linear-gradient(45deg, red, blue, green); */
  /* border: 3px solid transparent;
  border-radius: 12px;
  background-clip: padding-box, border-box;
  background-origin: padding-box, border-box;
  background-image: linear-gradient(to right, #222, #222), linear-gradient(90deg, #8F41E9, #578AEF); */
}

.row {
  display: flex;
}

.cell {
  /* border: 1px solid #E7DBDBFF; */
  box-sizing: border-box;
  cursor: pointer;
  border: 3px solid transparent;
  /* border-radius: 50%; */
  background-clip: padding-box, border-box;
  background-origin: padding-box, border-box;
  background-image: linear-gradient(to right, transparent, transparent), linear-gradient(90deg, rgba(0, 128, 0, 0.281), transparent);
}

.chess.empty {
  background-color: #181818;
}

.chess.black {
  background-size: cover;
  background-position: center;

  background-image: url('../../public/chess_black.png');
  box-sizing: border-box;
  cursor: pointer;

  border: 3px solid transparent;
  border-radius: 50%;
  background-clip: padding-box, border-box;
  background-origin: padding-box, border-box;
  background-image: linear-gradient(to right, #3d3a3a, #000000), linear-gradient(90deg, #8F41E9, #EE7DBBFF);
}

.chess.white {
  background-size: cover;
  background-position: center;

  background-image: url('../../public/chess_white.png');
  background-color: rgb(0, 0, 0);

  border: 3px solid transparent;
  border-radius: 50%;
  background-clip: padding-box, border-box;
  background-origin: padding-box, border-box;
  background-image: linear-gradient(to right, #BFBDCAFF, #F8EEEEFF), linear-gradient(90deg, #8EF37AFF, #40C5FAFF);
}
</style>
