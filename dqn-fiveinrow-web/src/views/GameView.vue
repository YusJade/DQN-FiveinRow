<template>
  <Board :board="game.getBoard()" :cell-click="handleCellClick" />
</template>

<script setup lang="ts">
import { ref } from 'vue';
import Board from '@/components/ChessBoard.vue';
import { Game } from '@/game';
import { AI } from '@/ai';

const game = ref(new Game(11));
const ai = ref(new AI(game.value.getBoard(), -1)); // AI 玩家

const handleCellClick = (row: number, col: number) => {
  if (game.value.makeMove(col, row)) {
    const winner = game.value.checkWin();
    if (winner !== 0) {
      alert(`Player ${winner} wins!`);
    }
  }
};
</script>
