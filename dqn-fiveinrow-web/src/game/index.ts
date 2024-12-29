export class Game {
  private board: number[][];
  private currentPlayer: number;

  constructor(size: number = 15) {
    this.board = Array(size).fill(null).map(() => Array(size).fill(0));
    this.currentPlayer = 1; // 1 为玩家，-1 为AI
  }

  // 获取当前棋盘状态
  public getBoard(): number[][] {
    return this.board;
  }

  // 玩家或AI下棋
  public makeMove(x: number, y: number): boolean {
    if (this.board[y][x] === 0) {
      this.board[y][x] = this.currentPlayer;
      this.currentPlayer = -this.currentPlayer; // 切换玩家
      return true;
    }
    return false;
  }

  // 判断胜负
  public checkWin(): number {
    // 这里只是一个简单的示范，你可以根据五子棋的规则完善
    const directions = [
      { dx: 0, dy: 1 },
      { dx: 1, dy: 0 },
      { dx: 1, dy: 1 },
      { dx: 1, dy: -1 },
    ];

    for (let y = 0; y < this.board.length; y++) {
      for (let x = 0; x < this.board[y].length; x++) {
        const player = this.board[y][x];
        if (player === 0) continue;

        for (let { dx, dy } of directions) {
          let count = 1;
          for (let i = 1; i < 5; i++) {
            const nx = x + dx * i;
            const ny = y + dy * i;
            if (nx >= 0 && ny >= 0 && nx < this.board.length && ny < this.board.length && this.board[ny][nx] === player) {
              count++;
            } else {
              break;
            }
          }
          if (count >= 5) {
            return player; // 当前玩家胜利
          }
        }
      }
    }

    return 0; // 没有胜者
  }
}
