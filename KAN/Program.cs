using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

var v1 = tensor(new double[,] { {0.1,0.2 },{ 0.3,0.4} });

v1.print();

testBBatch();


// See https://aka.ms/new-console-template for more information


void testBBatch()
{
    int numSpline = 5;
    int numSample = 100;
    int numGridInterval = 10;
    int k = 3;
    var x = torch.normal(0,1, new long[]{numSpline,numSample});
    Console.WriteLine($"x shape:["+string.Join(",", x.shape)+"]");
    var grids = torch.einsum("i,j->ij", torch.ones(new long[]{numSpline}), torch.linspace(-1,1, numGridInterval+1));
    Console.WriteLine($"grid shape:["+string.Join(",", grids.shape)+"]");
    var res = BBatch(x, grids, k);
    Console.WriteLine("result shape:[{"+string.Join(",", res.shape)+"]");
}
/**
    * @param x: Tensor of shape (x, y)
    * @param grid: Tensor of shape (x, z)
    * @param k: int
    * @param extend: bool
    * @return: Tensor of shape (x, z+k, y)
*/
Tensor BBatch(Tensor x, Tensor grid, int k = 0, bool extend = true)
{
     if(extend)
         grid = extendGrid(grid, k);

     grid = grid.unsqueeze(2);

     x = x.unsqueeze(1);

    Tensor value;
    if(k == 0) {
        using (var x1 = x.ge(grid.narrow(1, 0, grid.size(1) - 1)))
        using (var x2 = x.lt(grid.narrow(1, 1, grid.size(1) - 1)))
        {
            value = x1.mul(x2);//boolの行列になるのに注意
        }
    } else {
        var Bkm1 = BBatch(x.select(1,0), grid.select(2, 0),  k-1, false);
        value = (x - grid.narrow(1,0, grid.size(1) -(k+1))) / (grid.narrow(1,k,grid.size(1) - k - 1) -grid.narrow(1,0,grid.size(1) -k -1)) * Bkm1.narrow(1,0, Bkm1.size(1) - 1) +
               (grid.narrow(1,k+1,grid.size(1) - 1 - k) - x) / (grid.narrow(1,k+1,grid.size(1) - 1 - k) - grid.narrow(1,1,grid.size(1) - k - 1)) * Bkm1.narrow(1,1, Bkm1.size(1) - 1);
    }

    return value;

    Tensor extendGrid(Tensor grid, int kExtend = 0)
    {
        var h1 = grid.select(1,grid.size(1) -1);
        var h2 = grid.select(1,0);
        var h = (h1 - h2)/(grid.shape[1]-1);
       
        for(int i=0;i<kExtend;i++)
        {
            var newH = (grid.select(1,0) - h).view(-1,1);//[n]の行列を[n,1]に変換
            grid = torch.cat([newH, grid],1);
            newH = (grid.select(1, grid.size(1) -1) + h).view(-1,1);//[n]の行列を[n,1]に変換
            grid = torch.cat([grid, newH],1);
        }
        return grid;
    }
}

