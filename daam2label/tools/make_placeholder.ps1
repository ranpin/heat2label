Add-Type -AssemblyName System.Drawing

function Make-Scene {
    param([string]$Path, [string]$Subject, [int]$ObjX, [int]$ObjY, [int]$ObjW, [int]$ObjH, [string]$ObjColor)

    $W = 1792; $H = 1024
    $bmp = New-Object System.Drawing.Bitmap($W, $H)
    $g = [System.Drawing.Graphics]::FromImage($bmp)

    # Sky gradient
    $sky = [System.Drawing.Drawing2D.LinearGradientBrush]::new(
        [System.Drawing.Rectangle]::new(0,0,$W,[int]($H*0.55)),
        [System.Drawing.Color]::FromArgb(255,168,196,220),
        [System.Drawing.Color]::FromArgb(255,210,220,225),
        [System.Drawing.Drawing2D.LinearGradientMode]::Vertical)
    $g.FillRectangle($sky, 0, 0, $W, [int]($H*0.55))

    # Distant vegetation strip
    $g.FillRectangle([System.Drawing.Brushes]::DarkSeaGreen, 0, [int]($H*0.50), $W, [int]($H*0.10))

    # Ground / road
    $g.FillRectangle([System.Drawing.Brushes]::DimGray, 0, [int]($H*0.60), $W, [int]($H*0.40))
    # Perspective road (trapezoid)
    $pts = @(
        [System.Drawing.Point]::new([int]($W*0.30), $H),
        [System.Drawing.Point]::new([int]($W*0.47), [int]($H*0.60)),
        [System.Drawing.Point]::new([int]($W*0.53), [int]($H*0.60)),
        [System.Drawing.Point]::new([int]($W*0.70), $H)
    )
    $g.FillPolygon([System.Drawing.Brushes]::Gray, $pts)

    # Center dashes
    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::White, 6)
    $pen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
    $g.DrawLine($pen, [int]($W*0.50), [int]($H*0.62), [int]($W*0.50), $H)

    # Subject rectangle
    $col = [System.Drawing.Color]::FromName($ObjColor)
    $g.FillRectangle([System.Drawing.SolidBrush]::new($col), $ObjX, $ObjY, $ObjW, $ObjH)
    $g.DrawRectangle([System.Drawing.Pen]::new([System.Drawing.Color]::Black, 3), $ObjX, $ObjY, $ObjW, $ObjH)

    # Hood occlusion silhouette (fake vehicle hood at bottom)
    $hood = @(
        [System.Drawing.Point]::new(0, $H),
        [System.Drawing.Point]::new(0, [int]($H*0.93)),
        [System.Drawing.Point]::new([int]($W*0.30), [int]($H*0.90)),
        [System.Drawing.Point]::new([int]($W*0.70), [int]($H*0.90)),
        [System.Drawing.Point]::new($W, [int]($H*0.93)),
        [System.Drawing.Point]::new($W, $H)
    )
    $g.FillPolygon([System.Drawing.Brushes]::Black, $hood)

    # Label text
    $font = New-Object System.Drawing.Font("Arial", 28, [System.Drawing.FontStyle]::Bold)
    $g.DrawString("SYNTHETIC PLACEHOLDER - $Subject", $font, [System.Drawing.Brushes]::Yellow, 40, 30)

    $g.Dispose()
    $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Host "wrote $Path"
}

$ex = "e:\master\daam2label\examples"
Make-Scene -Path "$ex\motorcyclist_0001.png" -Subject "motorcyclist" -ObjX 980 -ObjY 700 -ObjW 140 -ObjH 180 -ObjColor "OrangeRed"
Make-Scene -Path "$ex\truck_0001.png"       -Subject "truck"       -ObjX 620 -ObjY 620 -ObjW 320 -ObjH 260 -ObjColor "SteelBlue"

"examples/motorcyclist_0001.png" | Set-Content "$ex\motorcyclist.txt"
"examples/truck_0001.png"        | Set-Content "$ex\truck.txt"
Write-Host "done"
